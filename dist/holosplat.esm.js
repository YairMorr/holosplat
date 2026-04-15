/*! HoloSplat v0.1.0 – WebGPU Gaussian Splat viewer | MIT */const SHADER=`

struct Uniforms {
  view       : mat4x4<f32>,
  proj       : mat4x4<f32>,
  viewport   : vec2<f32>,
  focal      : vec2<f32>,
  params     : vec4<f32>,  // .x = splatScale
};

struct Gaussian {
  pos   : vec3<f32>,
  color : vec4<f32>,
  scale : vec3<f32>,
  quat  : vec4<f32>,
};

@group(0) @binding(0) var<uniform>       uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2) var<storage, read> order     : array<u32>;

struct VOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) color         : vec4<f32>,
  @location(1) uv            : vec2<f32>,
  @location(2) conic         : vec3<f32>,
};

// Quaternion (xyzw) → column-major rotation matrix
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+w*z),       2.0*(x*z-w*y)),
    vec3<f32>(2.0*(x*y-w*z),         1.0 - 2.0*(x*x+z*z), 2.0*(y*z+w*x)),
    vec3<f32>(2.0*(x*z+w*y),         2.0*(y*z-w*x),       1.0 - 2.0*(x*x+y*y))
  );
}

// Degenerate vertex (moved outside clip space → triangle discarded)
fn degen() -> VOut {
  var o: VOut;
  o.clipPos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  o.color   = vec4<f32>(0.0);
  o.uv      = vec2<f32>(0.0);
  o.conic   = vec3<f32>(0.0);
  return o;
}

@vertex
fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VOut {
  // Quad corners (two triangles, CCW)
  const corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
  );

  let gi = order[ii];
  let g  = gaussians[gi];
  let corner = corners[vi];

  // ── Transform to view space ──────────────────────────────────────────────
  let viewPos4 = uniforms.view * vec4<f32>(g.pos, 1.0);
  let t = viewPos4.xyz;

  // Discard if behind near plane
  if t.z > -0.1 { return degen(); }

  // ── 3-D covariance ────────────────────────────────────────────────────────
  let splatScale = uniforms.params.x;
  let s = g.scale * splatScale;
  let R = quatToMat3(g.quat);
  // M = R * diag(s); Cov3D = M * Mᵀ
  let M    = mat3x3<f32>(R[0]*s.x, R[1]*s.y, R[2]*s.z);
  let cov3 = M * transpose(M);

  // ── Project covariance to 2-D ─────────────────────────────────────────────
  // Perspective Jacobian at view-space point t (camera looks down –Z)
  let tz2 = t.z * t.z;
  let fx  = uniforms.focal.x;
  let fy  = uniforms.focal.y;
  let J = mat3x3<f32>(
    vec3<f32>(fx / (-t.z),  0.0,           0.0),
    vec3<f32>(0.0,          fy / (-t.z),   0.0),
    vec3<f32>(fx*t.x / tz2, fy*t.y / tz2, 0.0)
  );
  // View rotation (upper-left 3×3 of view matrix, column-major)
  let W = mat3x3<f32>(
    uniforms.view[0].xyz,
    uniforms.view[1].xyz,
    uniforms.view[2].xyz
  );
  let T    = J * W;
  let cov2 = T * cov3 * transpose(T);

  // Extract 2×2 + low-pass filter (anti-aliasing)
  let a = cov2[0][0] + 0.3;   // Σ_xx  (cov2[col][row])
  let b = cov2[0][1];          // Σ_xy
  let c = cov2[1][1] + 0.3;   // Σ_yy

  // ── Inverse covariance (conic) ────────────────────────────────────────────
  let det = a*c - b*b;
  if det < 1e-4 { return degen(); }
  let inv = 1.0 / det;
  // conic = (A, B, C) such that Mahalanobis² = A·dx² + 2B·dx·dy + C·dy²
  let conic = vec3<f32>(c*inv, -b*inv, a*inv);

  // ── Bounding radius (3σ of largest eigenvalue) ────────────────────────────
  let mid    = 0.5 * (a + c);
  let disc   = sqrt(max(0.1, mid*mid - det));
  let radius = ceil(3.0 * sqrt(mid + disc));

  // ── Screen-space quad placement ───────────────────────────────────────────
  let clip    = uniforms.proj * viewPos4;
  let ndcXY   = clip.xy / clip.w;
  let pixOff  = corner * radius;
  let ndcOff  = pixOff / (uniforms.viewport * 0.5);

  var o: VOut;
  o.clipPos = vec4<f32>(ndcXY + ndcOff, clip.z / clip.w, 1.0);
  o.color   = g.color;
  o.uv      = pixOff;
  o.conic   = conic;
  return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let d = in.uv;
  let power = -0.5 * (in.conic.x*d.x*d.x + 2.0*in.conic.y*d.x*d.y + in.conic.z*d.y*d.y);
  if power > 0.0 { discard; }
  let alpha = in.color.a * exp(power);
  if alpha < 1.0/255.0 { discard; }
  // Premultiplied alpha output (blend: src=ONE, dst=ONE_MINUS_SRC_ALPHA)
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;class OrbitCamera{constructor({fov=60,near=0.1,far=2000}={}){this.fov=fov*Math.PI/180;this.near=near;this.far=far;this.theta=0;this.phi=0.2;this.radius=5;this.target=[0,0,0];this._drag=null;this._touches=[];this.viewMatrix=new Float32Array(16);this.projMatrix=new Float32Array(16);}
attach(canvas){this._canvas=canvas;this._onMouseDown=e=>this._mouseDown(e);this._onMouseMove=e=>this._mouseMove(e);this._onMouseUp=()=>{this._drag=null;};this._onWheel=e=>this._wheel(e);this._onTouchStart=e=>this._touchStart(e);this._onTouchMove=e=>this._touchMove(e);this._onTouchEnd=e=>this._touchEnd(e);this._onCtxMenu=e=>e.preventDefault();canvas.addEventListener('mousedown',this._onMouseDown);canvas.addEventListener('mousemove',this._onMouseMove);canvas.addEventListener('mouseup',this._onMouseUp);canvas.addEventListener('mouseleave',this._onMouseUp);canvas.addEventListener('wheel',this._onWheel,{passive:false});canvas.addEventListener('touchstart',this._onTouchStart,{passive:false});canvas.addEventListener('touchmove',this._onTouchMove,{passive:false});canvas.addEventListener('touchend',this._onTouchEnd);canvas.addEventListener('contextmenu',this._onCtxMenu);}
detach(){const c=this._canvas;if(!c)return;c.removeEventListener('mousedown',this._onMouseDown);c.removeEventListener('mousemove',this._onMouseMove);c.removeEventListener('mouseup',this._onMouseUp);c.removeEventListener('mouseleave',this._onMouseUp);c.removeEventListener('wheel',this._onWheel);c.removeEventListener('touchstart',this._onTouchStart);c.removeEventListener('touchmove',this._onTouchMove);c.removeEventListener('touchend',this._onTouchEnd);c.removeEventListener('contextmenu',this._onCtxMenu);this._canvas=null;}
_mouseDown(e){this._drag={x:e.clientX,y:e.clientY,button:e.button};e.preventDefault();}
_mouseMove(e){if(!this._drag)return;const dx=e.clientX-this._drag.x;const dy=e.clientY-this._drag.y;this._drag.x=e.clientX;this._drag.y=e.clientY;if(this._drag.button===2){this._pan(dx,dy);}else{this._orbit(dx,dy);}}
_wheel(e){e.preventDefault();const factor=e.deltaY>0?1.1:0.9;this.radius=Math.max(0.01,this.radius*factor);}
_touchStart(e){e.preventDefault();this._touches=Array.from(e.touches).map(t=>({id:t.identifier,x:t.clientX,y:t.clientY}));}
_touchMove(e){e.preventDefault();const prev=this._touches;const curr=Array.from(e.touches).map(t=>({id:t.identifier,x:t.clientX,y:t.clientY}));if(curr.length===1&&prev.length===1){const dx=curr[0].x-prev[0].x;const dy=curr[0].y-prev[0].y;this._orbit(dx,dy);}else if(curr.length===2&&prev.length===2){const prevDist=Math.hypot(prev[1].x-prev[0].x,prev[1].y-prev[0].y);const currDist=Math.hypot(curr[1].x-curr[0].x,curr[1].y-curr[0].y);if(prevDist>0){this.radius=Math.max(0.01,this.radius*(prevDist/currDist));}
const prevCx=(prev[0].x+prev[1].x)*0.5;const prevCy=(prev[0].y+prev[1].y)*0.5;const currCx=(curr[0].x+curr[1].x)*0.5;const currCy=(curr[0].y+curr[1].y)*0.5;this._pan(currCx-prevCx,currCy-prevCy);}
this._touches=curr;}
_touchEnd(e){this._touches=Array.from(e.touches).map(t=>({id:t.identifier,x:t.clientX,y:t.clientY}));}
_orbit(dx,dy){const speed=0.005;this.theta-=dx*speed;this.phi=Math.max(-Math.PI/2+0.01,Math.min(Math.PI/2-0.01,this.phi+dy*speed));}
_pan(dx,dy){const speed=this.radius*0.001;const right=this._cameraRight();const up=this._cameraUp();this.target[0]-=(right[0]*dx-up[0]*dy)*speed;this.target[1]-=(right[1]*dx-up[1]*dy)*speed;this.target[2]-=(right[2]*dx-up[2]*dy)*speed;}
_cameraRight(){return[this.viewMatrix[0],this.viewMatrix[4],this.viewMatrix[8]];}
_cameraUp(){return[this.viewMatrix[1],this.viewMatrix[5],this.viewMatrix[9]];}
update(width,height){const eye=this._eye();lookAt(eye,this.target,[0,1,0],this.viewMatrix);perspective(this.fov,width/height,this.near,this.far,this.projMatrix);}
_eye(){const cp=Math.cos(this.phi),sp=Math.sin(this.phi);const ct=Math.cos(this.theta),st=Math.sin(this.theta);return[this.target[0]+this.radius*cp*st,this.target[1]+this.radius*sp,this.target[2]+this.radius*cp*ct,];}
focalLength(height){return(height*0.5)/Math.tan(this.fov*0.5);}
setFromLookAt(eye,target){this.target=[target[0],target[1],target[2]];const dx=eye[0]-target[0];const dy=eye[1]-target[1];const dz=eye[2]-target[2];this.radius=Math.hypot(dx,dy,dz)||0.001;this.phi=Math.asin(Math.max(-1,Math.min(1,dy/this.radius)));this.theta=Math.atan2(dx,dz);}
fitScene(positions,numSplats){let minX=Infinity,minY=Infinity,minZ=Infinity;let maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;for(let i=0;i<numSplats;i++){const j=i*16;const x=positions[j],y=positions[j+1],z=positions[j+2];if(x<minX)minX=x;if(x>maxX)maxX=x;if(y<minY)minY=y;if(y>maxY)maxY=y;if(z<minZ)minZ=z;if(z>maxZ)maxZ=z;}
this.target=[(minX+maxX)*0.5,(minY+maxY)*0.5,(minZ+maxZ)*0.5,];const extent=Math.max(maxX-minX,maxY-minY,maxZ-minZ)*0.5;this.radius=extent/Math.tan(this.fov*0.5)*1.2;}}
function lookAt(eye,center,up,out){const[ex,ey,ez]=eye;const[cx,cy,cz]=center;const[ux,uy,uz]=up;let zx=ex-cx,zy=ey-cy,zz=ez-cz;let zl=Math.hypot(zx,zy,zz);zx/=zl;zy/=zl;zz/=zl;let xx=uy*zz-uz*zy;let xy=uz*zx-ux*zz;let xz=ux*zy-uy*zx;const xl=Math.hypot(xx,xy,xz);xx/=xl;xy/=xl;xz/=xl;const yx=zy*xz-zz*xy;const yy=zz*xx-zx*xz;const yz=zx*xy-zy*xx;out[0]=xx;out[1]=yx;out[2]=zx;out[3]=0;out[4]=xy;out[5]=yy;out[6]=zy;out[7]=0;out[8]=xz;out[9]=yz;out[10]=zz;out[11]=0;out[12]=-(xx*ex+xy*ey+xz*ez);out[13]=-(yx*ex+yy*ey+yz*ez);out[14]=-(zx*ex+zy*ey+zz*ez);out[15]=1;}
function perspective(fovY,aspect,near,far,out){const f=1.0/Math.tan(fovY*0.5);const nf=near-far;out[0]=f/aspect;out[1]=0;out[2]=0;out[3]=0;out[4]=0;out[5]=f;out[6]=0;out[7]=0;out[8]=0;out[9]=0;out[10]=far/nf;out[11]=-1;out[12]=0;out[13]=0;out[14]=near*far/nf;out[15]=0;}
function createSorter(maxN){const keys0=new Uint16Array(maxN);const keys1=new Uint16Array(maxN);const idx0=new Uint32Array(maxN);const idx1=new Uint32Array(maxN);const counts=new Int32Array(256);const pfx=new Int32Array(256);return function sort(depths,N){let minD=depths[0],maxD=depths[0];for(let i=1;i<N;i++){if(depths[i]<minD)minD=depths[i];if(depths[i]>maxD)maxD=depths[i];}
const range=maxD-minD;const scale=range>0?65535/range:0;for(let i=0;i<N;i++){keys0[i]=Math.round((depths[i]-minD)*scale)|0;idx0[i]=i;}
counts.fill(0);for(let i=0;i<N;i++)counts[keys0[i]&0xff]++;pfx[0]=0;for(let i=1;i<256;i++)pfx[i]=pfx[i-1]+counts[i-1];for(let i=0;i<N;i++){const k=keys0[i]&0xff;const pos=pfx[k]++;keys1[pos]=keys0[i];idx1[pos]=idx0[i];}
counts.fill(0);for(let i=0;i<N;i++)counts[(keys1[i]>>8)&0xff]++;pfx[0]=0;for(let i=1;i<256;i++)pfx[i]=pfx[i-1]+counts[i-1];for(let i=0;i<N;i++){const k=(keys1[i]>>8)&0xff;const pos=pfx[k]++;idx0[pos]=idx1[i];}
return idx0;};}
async function fetchWithProgress(url,onProgress){const res=await fetch(url);if(!res.ok)throw new Error(`HTTP ${res.status} loading ${url}`);const total=parseInt(res.headers.get('content-length')||'0',10);const reader=res.body.getReader();const chunks=[];let loaded=0;for(;;){const{done,value}=await reader.read();if(done)break;chunks.push(value);loaded+=value.byteLength;if(onProgress&&total>0)onProgress(loaded/total);}
const combined=new Uint8Array(loaded);let offset=0;for(const chunk of chunks){combined.set(chunk,offset);offset+=chunk.byteLength;}
return combined.buffer;}
async function loadSplat(url,onProgress){const buffer=await fetchWithProgress(url,onProgress);return parseSplat(buffer);}
function parseSplat(buffer){const STRIDE=32;const N=Math.floor(buffer.byteLength/STRIDE);if(N===0)throw new Error('Empty or invalid .splat file');const src=new DataView(buffer);const data=new Float32Array(N*16);for(let i=0;i<N;i++){const s=i*STRIDE;const d=i*16;data[d+0]=src.getFloat32(s+0,true);data[d+1]=src.getFloat32(s+4,true);data[d+2]=src.getFloat32(s+8,true);data[d+4]=src.getUint8(s+24)/255;data[d+5]=src.getUint8(s+25)/255;data[d+6]=src.getUint8(s+26)/255;data[d+7]=src.getUint8(s+27)/255;data[d+8]=src.getFloat32(s+12,true);data[d+9]=src.getFloat32(s+16,true);data[d+10]=src.getFloat32(s+20,true);const qw=(src.getUint8(s+28)-128)/128;const qx=(src.getUint8(s+29)-128)/128;const qy=(src.getUint8(s+30)-128)/128;const qz=(src.getUint8(s+31)-128)/128;const ql=Math.hypot(qx,qy,qz,qw)||1;data[d+12]=qx/ql;data[d+13]=qy/ql;data[d+14]=qz/ql;data[d+15]=qw/ql;}
return{data,count:N};}
const SH_C0=0.28209479177387814;async function loadPly(url,onProgress){const buffer=await fetchWithProgress(url,onProgress);return parsePly(buffer);}
function parsePly(buffer){const{numVertices,properties,dataOffset}=parseHeader(buffer);if(numVertices===0)throw new Error('PLY file contains no vertices');const propMap={};let stride=0;for(const p of properties){propMap[p.name]={offset:stride,type:p.type};stride+=sizeOf(p.type);}
const required=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','opacity'];for(const r of required){if(!propMap[r])throw new Error(`PLY missing required property: ${r}`);}
const hasColor=propMap['f_dc_0']&&propMap['f_dc_1']&&propMap['f_dc_2'];const src=new DataView(buffer,dataOffset);const data=new Float32Array(numVertices*16);for(let i=0;i<numVertices;i++){const base=i*stride;const d=i*16;data[d+0]=read(src,base,propMap['x']);data[d+1]=read(src,base,propMap['y']);data[d+2]=read(src,base,propMap['z']);if(hasColor){data[d+4]=clamp01(0.5+SH_C0*read(src,base,propMap['f_dc_0']));data[d+5]=clamp01(0.5+SH_C0*read(src,base,propMap['f_dc_1']));data[d+6]=clamp01(0.5+SH_C0*read(src,base,propMap['f_dc_2']));}else{data[d+4]=1;data[d+5]=1;data[d+6]=1;}
data[d+7]=sigmoid(read(src,base,propMap['opacity']));data[d+8]=Math.exp(read(src,base,propMap['scale_0']));data[d+9]=Math.exp(read(src,base,propMap['scale_1']));data[d+10]=Math.exp(read(src,base,propMap['scale_2']));const rw=read(src,base,propMap['rot_0']);const rx=read(src,base,propMap['rot_1']);const ry=read(src,base,propMap['rot_2']);const rz=read(src,base,propMap['rot_3']);const rl=Math.hypot(rx,ry,rz,rw)||1;data[d+12]=rx/rl;data[d+13]=ry/rl;data[d+14]=rz/rl;data[d+15]=rw/rl;}
return{data,count:numVertices};}
function parseHeader(buffer){const bytes=new Uint8Array(buffer);const END_TAG='end_header';let headerEnd=-1;let text='';for(let i=0;i<bytes.length;i++){text+=String.fromCharCode(bytes[i]);if(text.endsWith(END_TAG)){headerEnd=i+1;if(bytes[headerEnd]===13)headerEnd++;if(bytes[headerEnd]===10)headerEnd++;break;}}
if(headerEnd<0)throw new Error('Invalid PLY: end_header not found');const lines=text.split('\n');let numVertices=0;const properties=[];let inVertex=false;for(const raw of lines){const line=raw.trim();const parts=line.split(/\s+/);if(parts[0]==='element'){inVertex=parts[1]==='vertex';if(inVertex)numVertices=parseInt(parts[2],10);}else if(parts[0]==='property'&&inVertex){properties.push({type:parts[1],name:parts[2]});}}
return{numVertices,properties,dataOffset:headerEnd};}
function sizeOf(type){switch(type){case'float':case'float32':case'int':case'uint':return 4;case'double':case'int64':case'uint64':return 8;case'short':case'ushort':case'int16':case'uint16':return 2;case'char':case'uchar':case'int8':case'uint8':return 1;default:return 4;}}
function read(view,base,prop){const off=base+prop.offset;switch(prop.type){case'float':case'float32':return view.getFloat32(off,true);case'double':return view.getFloat64(off,true);case'int':case'int32':return view.getInt32(off,true);case'uint':case'uint32':return view.getUint32(off,true);case'short':case'int16':return view.getInt16(off,true);case'ushort':case'uint16':return view.getUint16(off,true);case'char':case'int8':return view.getInt8(off);case'uchar':case'uint8':return view.getUint8(off);default:return view.getFloat32(off,true);}}
function sigmoid(x){return 1/(1+Math.exp(-x));}
function clamp01(x){return x<0?0:x>1?1:x;}
const MAGIC=0x5053474E;const SQRT2INV=1/Math.SQRT2;async function loadSpz(url,onProgress){const compressed=await fetchWithProgress(url,onProgress);const buffer=await decompressGzip(compressed);return parseSpz(buffer);}
function parseSpz(buffer){const view=new DataView(buffer);const magic=view.getUint32(0,true);const version=view.getUint32(4,true);const numPoints=view.getUint32(8,true);const shDegree=view.getUint8(12);const fractionalBits=view.getUint8(13);if(magic!==MAGIC){throw new Error(`Invalid .spz magic: 0x${magic.toString(16).toUpperCase()} `+`(expected 0x${MAGIC.toString(16).toUpperCase()})`);}
if(version<2||version>4){console.warn(`HoloSplat: .spz version ${version} is untested; attempting load anyway`);}
const posDiv=1<<fractionalBits;const rotSize=version>=3?4:3;const offPos=16;const offAlpha=offPos+numPoints*9;const offColor=offAlpha+numPoints*1;const offScale=offColor+numPoints*3;const offRot=offScale+numPoints*3;const data=new Float32Array(numPoints*16);for(let i=0;i<numPoints;i++){const d=i*16;const pb=offPos+i*9;data[d+0]=readInt24(view,pb+0)/posDiv;data[d+1]=readInt24(view,pb+3)/posDiv;data[d+2]=readInt24(view,pb+6)/posDiv;const cb=offColor+i*3;data[d+4]=view.getUint8(cb+0)/255;data[d+5]=view.getUint8(cb+1)/255;data[d+6]=view.getUint8(cb+2)/255;data[d+7]=view.getUint8(offAlpha+i)/255;const sb=offScale+i*3;data[d+8]=Math.exp((view.getUint8(sb+0)-128)/16);data[d+9]=Math.exp((view.getUint8(sb+1)-128)/16);data[d+10]=Math.exp((view.getUint8(sb+2)-128)/16);const rb=offRot+i*rotSize;let qx,qy,qz,qw;if(version>=3){const u32=view.getUint32(rb,true);const idx=u32&3;const s=SQRT2INV/512;const a=signExtend10((u32>>2)&0x3FF)*s;const b_=signExtend10((u32>>12)&0x3FF)*s;const c=signExtend10((u32>>22)&0x3FF)*s;const d_=Math.sqrt(Math.max(0,1-a*a-b_*b_-c*c));switch(idx){case 0:qx=d_;qy=a;qz=b_;qw=c;break;case 1:qx=a;qy=d_;qz=b_;qw=c;break;case 2:qx=a;qy=b_;qz=d_;qw=c;break;default:qx=a;qy=b_;qz=c;qw=d_;}}else{qx=view.getInt8(rb+0)/128;qy=view.getInt8(rb+1)/128;qz=view.getInt8(rb+2)/128;qw=Math.sqrt(Math.max(0,1-qx*qx-qy*qy-qz*qz));}
const ql=Math.hypot(qx,qy,qz,qw)||1;data[d+12]=qx/ql;data[d+13]=qy/ql;data[d+14]=qz/ql;data[d+15]=qw/ql;}
return{data,count:numPoints};}
function readInt24(view,offset){const lo=view.getUint8(offset);const mi=view.getUint8(offset+1);const hi=view.getInt8(offset+2);return lo|(mi<<8)|(hi<<16);}
function signExtend10(v){return v&0x200?v|0xFFFFFC00:v;}
async function decompressGzip(buffer){if(typeof DecompressionStream==='undefined'){throw new Error('DecompressionStream is not available in this environment');}
const stream=new DecompressionStream('gzip');const writer=stream.writable.getWriter();writer.write(buffer);writer.close();return new Response(stream.readable).arrayBuffer();}
class Animation{constructor(data){if(!data.frames||data.frames.length===0){throw new Error('HoloSplat Animation: no frame data');}
this.fps=data.fps??24;this.frameCount=data.frameCount??Math.floor(data.frames.length/6);this.fov=data.fov??null;this.callouts=data.callouts??[];this.loop=true;this._frames=new Float32Array(data.frames);this._time=0;this._playing=true;}
get duration(){return this.frameCount/this.fps;}
get time(){return this._time;}
get playing(){return this._playing;}
play(){this._playing=true;}
pause(){this._playing=false;}
seek(seconds){this._time=Math.max(0,Math.min(this.duration,seconds));}
tick(dt){if(!this._playing)return;this._time+=dt;if(this._time>=this.duration){this._time=this.loop?this._time%this.duration:this.duration;if(!this.loop)this._playing=false;}}
getCameraFrame(){const frame=Math.min(Math.floor(this._time*this.fps),this.frameCount-1);const i=frame*6;const f=this._frames;return{eye:[f[i],f[i+1],f[i+2]],target:[f[i]+f[i+3],f[i+1]+f[i+4],f[i+2]+f[i+5]],};}}
async function loadAnimation(url){const res=await fetch(url);if(!res.ok)throw new Error(`HoloSplat: failed to load animation "${url}" (HTTP ${res.status})`);return new Animation(await res.json());}
const SPZ_MAGIC=0x5053474E;const SPZ_VERSION=3;async function compressToSpz(data,count,opts={}){const raw=encodeSpz(data,count,opts);return compressGzip(raw);}
function encodeSpz(data,count,opts={}){let{fractionalBits}=opts;if(fractionalBits==null){let maxAbsPos=0;for(let i=0;i<count;i++){const b=i*16;if(Math.abs(data[b])>maxAbsPos)maxAbsPos=Math.abs(data[b]);if(Math.abs(data[b+1])>maxAbsPos)maxAbsPos=Math.abs(data[b+1]);if(Math.abs(data[b+2])>maxAbsPos)maxAbsPos=Math.abs(data[b+2]);}
const INT24_MAX=(1<<23)-1;fractionalBits=maxAbsPos>0?Math.min(20,Math.max(0,Math.floor(Math.log2(INT24_MAX/maxAbsPos)))):12;}
const HEADER=16;const total=HEADER+count*(9+1+3+3+4);const buf=new ArrayBuffer(total);const view=new DataView(buf);const u8=new Uint8Array(buf);view.setUint32(0,SPZ_MAGIC,true);view.setUint32(4,SPZ_VERSION,true);view.setUint32(8,count,true);view.setUint8(12,0);view.setUint8(13,fractionalBits);view.setUint8(14,0);view.setUint8(15,0);const offPos=HEADER;const offAlpha=offPos+count*9;const offColor=offAlpha+count*1;const offScale=offColor+count*3;const offRot=offScale+count*3;const posScale=1<<fractionalBits;for(let i=0;i<count;i++){const d=i*16;writeInt24(view,offPos+i*9+0,data[d+0]*posScale);writeInt24(view,offPos+i*9+3,data[d+1]*posScale);writeInt24(view,offPos+i*9+6,data[d+2]*posScale);u8[offAlpha+i]=clampU8(data[d+7]*255);u8[offColor+i*3+0]=clampU8(data[d+4]*255);u8[offColor+i*3+1]=clampU8(data[d+5]*255);u8[offColor+i*3+2]=clampU8(data[d+6]*255);u8[offScale+i*3+0]=encodeScale(data[d+8]);u8[offScale+i*3+1]=encodeScale(data[d+9]);u8[offScale+i*3+2]=encodeScale(data[d+10]);view.setUint32(offRot+i*4,encodeQuat(data[d+12],data[d+13],data[d+14],data[d+15]),true);}
return u8;}
function writeInt24(view,offset,value){const v=Math.max(-8388608,Math.min(8388607,Math.round(value)));view.setUint8(offset,v&0xFF);view.setUint8(offset+1,(v>>8)&0xFF);view.setUint8(offset+2,(v>>16)&0xFF);}
function clampU8(v){return Math.max(0,Math.min(255,Math.round(v)));}
function encodeScale(linear){return clampU8(Math.log(Math.max(1e-9,linear))*16+128);}
function encodeQuat(qx,qy,qz,qw){const len=Math.hypot(qx,qy,qz,qw)||1;const q=[qx/len,qy/len,qz/len,qw/len];let maxIdx=0;for(let j=1;j<4;j++){if(Math.abs(q[j])>Math.abs(q[maxIdx]))maxIdx=j;}
const sign=q[maxIdx]<0?-1:1;const others=[0,1,2,3].filter(j=>j!==maxIdx);const S=512*Math.SQRT2;const encode10=j=>Math.max(-512,Math.min(511,Math.round(q[j]*sign*S)));const a=encode10(others[0]);const b=encode10(others[1]);const c=encode10(others[2]);return((maxIdx&3)|((a&0x3FF)<<2)|((b&0x3FF)<<12)|((c&0x3FF)<<22))>>>0;}
async function compressGzip(data){if(typeof CompressionStream==='undefined'){throw new Error('CompressionStream API is not available in this environment');}
const stream=new CompressionStream('gzip');const writer=stream.writable.getWriter();writer.write(data);writer.close();return new Response(stream.readable).arrayBuffer();}
const U_VIEW=0;const U_PROJ=16;const U_VIEWPORT=32;const U_FOCAL=34;const U_PARAMS=36;const U_SIZE=40;class Renderer{constructor(canvas,background){this.canvas=canvas;this.background=parseBackground(background);this.device=null;this.context=null;this.pipeline=null;this.bindGroup=null;this._uniformBuf=null;this._gaussianBuf=null;this._orderBuf=null;this._uniforms=new Float32Array(U_SIZE);this._uniforms[U_PARAMS]=1.0;this._numSplats=0;}
async init(){if(!navigator.gpu)throw new Error('WebGPU is not supported in this browser.');const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter found.');this.device=await adapter.requestDevice();this.context=this.canvas.getContext('webgpu');this._format=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:this._format,alphaMode:'premultiplied',});this._createPipeline();this._uniformBuf=this._createBuffer(U_SIZE*4,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);}
uploadGaussians(data,count){this._numSplats=count;if(this._gaussianBuf)this._gaussianBuf.destroy();this._gaussianBuf=this.device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,});this.device.queue.writeBuffer(this._gaussianBuf,0,data);if(this._orderBuf)this._orderBuf.destroy();this._orderBuf=this.device.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,});this._rebuildBindGroup();}
updateUniforms({view,proj,width,height,focal}){const u=this._uniforms;u.set(view,U_VIEW);u.set(proj,U_PROJ);u[U_VIEWPORT]=width;u[U_VIEWPORT+1]=height;u[U_FOCAL]=focal;u[U_FOCAL+1]=focal;this.device.queue.writeBuffer(this._uniformBuf,0,u);}
updateOrder(sortedIndices,count){this.device.queue.writeBuffer(this._orderBuf,0,sortedIndices.buffer,0,count*4);}
setSplatScale(s){this._uniforms[U_PARAMS]=s;}
setBackground(bg){this.background=parseBackground(bg);}
draw(){if(!this._numSplats||!this.bindGroup)return;const encoder=this.device.createCommandEncoder();const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:this.background,loadOp:'clear',storeOp:'store',}],});pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bindGroup);pass.draw(6,this._numSplats,0,0);pass.end();this.device.queue.submit([encoder.finish()]);}
destroy(){this._uniformBuf?.destroy();this._gaussianBuf?.destroy();this._orderBuf?.destroy();this.context?.unconfigure();}
_createPipeline(){const module=this.device.createShaderModule({code:SHADER});this.pipeline=this.device.createRenderPipeline({layout:'auto',vertex:{module,entryPoint:'vs_main'},fragment:{module,entryPoint:'fs_main',targets:[{format:this._format,blend:{color:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'},},}],},primitive:{topology:'triangle-list',cullMode:'none'},});}
_createBuffer(size,usage){return this.device.createBuffer({size,usage});}
_rebuildBindGroup(){this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._uniformBuf}},{binding:1,resource:{buffer:this._gaussianBuf}},{binding:2,resource:{buffer:this._orderBuf}},],});}}
function parseBackground(bg){if(!bg||bg==='transparent')return{r:0,g:0,b:0,a:0};if(Array.isArray(bg))return{r:bg[0],g:bg[1],b:bg[2],a:bg[3]??1};if(typeof bg==='string'){const hex=bg.replace('#','');if(hex.length===6){return{r:parseInt(hex.slice(0,2),16)/255,g:parseInt(hex.slice(2,4),16)/255,b:parseInt(hex.slice(4,6),16)/255,a:1,};}
if(hex.length===8){return{r:parseInt(hex.slice(0,2),16)/255,g:parseInt(hex.slice(2,4),16)/255,b:parseInt(hex.slice(4,6),16)/255,a:parseInt(hex.slice(6,8),16)/255,};}}
return{r:0,g:0,b:0,a:1};}
class Viewer{constructor(options={}){const{canvas,background='#000000',fov=60,near=0.1,far=2000,splatScale=1.0,autoRotate=false,onProgress,onError,}=options;this._canvas=resolveCanvas(canvas);this._onProgress=onProgress;this._onError=onError;this._autoRotate=autoRotate;this._splatScale=splatScale;this._renderer=new Renderer(this._canvas,background);this._camera=new OrbitCamera({fov,near,far});this._gaussians=null;this._numSplats=0;this._depths=null;this._sort=null;this._rafId=null;this._running=false;this._resizeObs=null;this._animation=null;this._lastTick=null;this.onFrame=null;}
async init(){await this._renderer.init();this._renderer.setSplatScale(this._splatScale);this._camera.attach(this._canvas);this._observeResize();this._updateSize();}
async load(url){const ext=url.split('?')[0].split('.').pop().toLowerCase();const loaders={ply:loadPly,spz:loadSpz};const loader=loaders[ext]??loadSplat;const{data,count}=await loader(url,p=>{if(this._onProgress)this._onProgress(p);});this._gaussians=data;this._numSplats=count;this._depths=new Float32Array(count);this._sort=createSorter(count);this._renderer.uploadGaussians(data,count);this._camera.fitScene(data,count);}
start(){if(this._running)return;this._running=true;this._tick();}
stop(){this._running=false;if(this._rafId)cancelAnimationFrame(this._rafId);this._rafId=null;}
destroy(){this.stop();this._camera.detach();this._renderer.destroy();this._resizeObs?.disconnect();}
setBackground(bg){this._renderer.setBackground(bg);}
setSplatScale(s){this._splatScale=s;this._renderer.setSplatScale(s);}
setAutoRotate(v){this._autoRotate=v;}
resetCamera(){this._camera.fitScene(this._gaussians,this._numSplats);}
setAnimation(anim){this._animation=anim;if(anim?.fov!=null){this._camera.fov=anim.fov*Math.PI/180;}}
async loadAnimationUrl(url){const anim=await loadAnimation(url);this.setAnimation(anim);return anim;}
projectCallouts(callouts){const view=this._camera.viewMatrix;const proj=this._camera.projMatrix;const w=this._canvas.width;const h=this._canvas.height;const out=[];for(const c of callouts){const[px,py,pz]=c.pos;const vx=view[0]*px+view[4]*py+view[8]*pz+view[12];const vy=view[1]*px+view[5]*py+view[9]*pz+view[13];const vz=view[2]*px+view[6]*py+view[10]*pz+view[14];if(vz>=0){out.push({id:c.id,visible:false,x:0,y:0});continue;}
const cw=-vz;const sx=(proj[0]*vx/cw*0.5+0.5)*w;const sy=(1-(proj[5]*vy/cw*0.5+0.5))*h;out.push({id:c.id,visible:true,x:sx,y:sy});}
return out;}
get camera(){return this._camera;}
_tick(){if(!this._running)return;this._rafId=requestAnimationFrame(()=>this._tick());const now=performance.now();const dt=this._lastTick?Math.min((now-this._lastTick)/1000,0.1):0;this._lastTick=now;const w=this._canvas.width;const h=this._canvas.height;if(this._animation){this._animation.tick(dt);const{eye,target}=this._animation.getCameraFrame();this._camera.setFromLookAt(eye,target);}else if(this._autoRotate){this._camera.theta+=0.005;}
if(!this._numSplats)return;this._camera.update(w,h);if(this.onFrame){this.onFrame(this._camera.viewMatrix,this._camera.projMatrix,w,h);}
const view=this._camera.viewMatrix;const proj=this._camera.projMatrix;const focal=this._camera.focalLength(h);this._computeDepths(view);const order=this._sort(this._depths,this._numSplats);this._renderer.updateUniforms({view,proj,width:w,height:h,focal});this._renderer.updateOrder(order,this._numSplats);this._renderer.draw();}
_computeDepths(view){const v0=view[2],v1=view[6],v2=view[10],v3=view[14];const gs=this._gaussians;const dep=this._depths;const N=this._numSplats;for(let i=0;i<N;i++){const j=i*16;dep[i]=v0*gs[j]+v1*gs[j+1]+v2*gs[j+2]+v3;}}
_observeResize(){if(typeof ResizeObserver==='undefined')return;this._resizeObs=new ResizeObserver(()=>this._updateSize());this._resizeObs.observe(this._canvas);}
_updateSize(){const dpr=window.devicePixelRatio||1;const w=Math.round(this._canvas.clientWidth*dpr);const h=Math.round(this._canvas.clientHeight*dpr);if(w&&h&&(this._canvas.width!==w||this._canvas.height!==h)){this._canvas.width=w;this._canvas.height=h;}}}
function resolveCanvas(canvas){if(!canvas)throw new Error('HoloSplat: canvas option is required');if(typeof canvas==='string'){const el=document.querySelector(canvas);if(!el)throw new Error(`HoloSplat: canvas selector "${canvas}" not found`);return el;}
return canvas;}
const PLAYER_CSS=`
.hs-player{position:relative;overflow:hidden;}
.hs-player canvas{position:absolute;inset:0;width:100%;height:100%;display:block;}
.hs-player .hs-overlay{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;
  pointer-events:none;font-family:system-ui,sans-serif;
}
.hs-player .hs-spinner{
  width:24px;height:24px;border-radius:50%;
  border:2px solid rgba(255,255,255,.15);border-top-color:#3a7aff;
  animation:hs-spin .7s linear infinite;
}
@keyframes hs-spin{to{transform:rotate(360deg);}}
.hs-player .hs-bar-wrap{width:140px;height:3px;background:rgba(255,255,255,.1);border-radius:2px;overflow:hidden;}
.hs-player .hs-bar{height:100%;background:#3a7aff;width:0%;transition:width .1s;}
.hs-player .hs-msg{font-size:.78rem;color:rgba(255,255,255,.45);text-align:center;max-width:260px;line-height:1.5;padding:0 16px;}
.hs-player .hs-msg.hs-err{color:#f87;}
.hs-player .hs-callouts{position:absolute;inset:0;pointer-events:none;}
.hs-callout{position:absolute;pointer-events:auto;transform:translate(-50%,-50%);}
.hs-callout--hidden{display:none;}
`;let cssInjected=false;function injectCss(){if(cssInjected||typeof document==='undefined')return;cssInjected=true;const s=document.createElement('style');s.textContent=PLAYER_CSS;document.head.appendChild(s);}
function player(container,opts={}){injectCss();const root=typeof container==='string'?document.querySelector(container):container;if(!root)throw new Error(`HoloSplat: container not found — "${container}"`);const{src,animation:animSrc,background='transparent',fov=60,near=0.1,far=2000,splatScale=1,autoRotate=false,onLoad,onProgress,onError,}=opts;root.classList.add('hs-player');const canvas=document.createElement('canvas');const calloutEl=document.createElement('div');calloutEl.className='hs-callouts';const overlay=document.createElement('div');overlay.className='hs-overlay';overlay.innerHTML='<div class="hs-spinner"></div>'+'<div class="hs-bar-wrap"><div class="hs-bar"></div></div>'+'<div class="hs-msg"></div>';root.appendChild(canvas);root.appendChild(calloutEl);root.appendChild(overlay);const spinner=overlay.querySelector('.hs-spinner');const barWrap=overlay.querySelector('.hs-bar-wrap');const bar=overlay.querySelector('.hs-bar');const msgEl=overlay.querySelector('.hs-msg');function showLoading(){spinner.style.display='';barWrap.style.display='';msgEl.textContent='';msgEl.className='hs-msg';overlay.style.display='flex';}
function showReady(){overlay.style.display='none';}
function showError(text){spinner.style.display='none';barWrap.style.display='none';msgEl.textContent=text;msgEl.className='hs-msg hs-err';overlay.style.display='flex';}
overlay.style.display='none';const viewer=new Viewer({canvas,background,fov,near,far,splatScale,autoRotate,onProgress:p=>{bar.style.width=`${(p * 100).toFixed(0)}%`;if(onProgress)onProgress(p);},});const calloutDivs={};function buildCallouts(callouts){calloutEl.innerHTML='';for(const key of Object.keys(calloutDivs))delete calloutDivs[key];for(const c of callouts){const div=document.createElement('div');div.className='hs-callout';div.dataset.id=c.id;calloutEl.appendChild(div);calloutDivs[c.id]=div;}}
viewer.onFrame=(_view,_proj,_w,_h)=>{if(!viewer._animation?.callouts.length)return;const projected=viewer.projectCallouts(viewer._animation.callouts);for(const{id,visible,x,y}of projected){const div=calloutDivs[id];if(!div)continue;if(visible){div.classList.remove('hs-callout--hidden');div.style.left=x+'px';div.style.top=y+'px';}else{div.classList.add('hs-callout--hidden');}}};async function load(url){showLoading();bar.style.width='0%';try{await viewer.load(url);showReady();if(onLoad)onLoad();}catch(err){const msg=navigator.gpu?err.message:'WebGPU not supported. Use Chrome 113+ or Edge 113+.';showError(msg);if(onError)onError(err);}}
async function loadAnim(url){try{const anim=await viewer.loadAnimationUrl(url);buildCallouts(anim.callouts);return anim;}catch(err){if(onError)onError(err);else throw err;}}
viewer.init().then(()=>{viewer.start();const loads=[];if(src)loads.push(load(src));if(animSrc)loads.push(loadAnim(animSrc));return Promise.all(loads);}).catch(err=>{const msg=navigator.gpu?err.message:'WebGPU not supported. Use Chrome 113+ or Edge 113+.';showError(msg);if(onError)onError(err);});return{load,loadAnim,destroy(){viewer.destroy();root.innerHTML='';root.classList.remove('hs-player');},setBackground(bg){viewer.setBackground(bg);},setSplatScale(s){viewer.setSplatScale(s);},setAutoRotate(v){viewer.setAutoRotate(v);},resetCamera(){viewer.resetCamera();},callout(id){return calloutDivs[id]??null;},get camera(){return viewer.camera;},get animation(){return viewer._animation;},};}
function autoInit(){document.querySelectorAll('[data-holosplat]').forEach(el=>{if(el._hsPlayer)return;const src=el.getAttribute('data-holosplat')||undefined;const anim=el.getAttribute('data-holosplat-anim')||undefined;el._hsPlayer=player(el,{src,animation:anim});});}
if(typeof document!=='undefined'){if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',autoInit);}else{autoInit();}}
async function create(options={}){const{onLoad,onError,src,...viewerOpts}=options;const viewer=new Viewer({...viewerOpts});const noop={destroy(){},setBackground(){},setSplatScale(){},setAutoRotate(){},resetCamera(){},camera:null,};try{await viewer.init();if(src)await viewer.load(src);}catch(err){viewer.destroy();if(onError){onError(err);return noop;}
throw err;}
viewer.start();onLoad?.();return{destroy(){viewer.destroy();},setBackground(bg){viewer.setBackground(bg);},setSplatScale(s){viewer.setSplatScale(s);},setAutoRotate(v){viewer.setAutoRotate(v);},resetCamera(){viewer.resetCamera();},get camera(){return viewer.camera;},};}
export{create,player,Viewer,Animation,loadAnimation,compressToSpz,encodeSpz,parseSplat,parsePly};