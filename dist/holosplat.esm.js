var xt=`

struct Uniforms {
  view       : mat4x4<f32>,
  proj       : mat4x4<f32>,
  viewport   : vec2<f32>,
  focal      : vec2<f32>,
  params     : vec4<f32>,  // .x = splatScale  .y = near (view-space units)
};

struct Gaussian {
  pos   : vec3<f32>,
  part  : f32,             // part index (0.0, 1.0, 2.0\u2026) \u2014 cast to u32 in shader
  color : vec4<f32>,
  scale : vec3<f32>,
  quat  : vec4<f32>,
};

@group(0) @binding(0) var<uniform>       uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read> gaussians  : array<Gaussian>;
@group(0) @binding(2) var<storage, read> order      : array<u32>;
@group(0) @binding(3) var<storage, read> transforms : array<mat4x4<f32>>;

struct VOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) color         : vec4<f32>,
  @location(1) uv            : vec2<f32>,
  @location(2) conic         : vec3<f32>,
};

// Quaternion (xyzw) \u2192 column-major rotation matrix
fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+w*z),       2.0*(x*z-w*y)),
    vec3<f32>(2.0*(x*y-w*z),         1.0 - 2.0*(x*x+z*z), 2.0*(y*z+w*x)),
    vec3<f32>(2.0*(x*z+w*y),         2.0*(y*z-w*x),       1.0 - 2.0*(x*x+y*y))
  );
}

// Degenerate vertex (moved outside clip space \u2192 triangle discarded)
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

  // \u2500\u2500 Part transform: local \u2192 world \u2192 view \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let partMat  = transforms[u32(g.part)];
  let viewPos4 = uniforms.view * (partMat * vec4<f32>(g.pos, 1.0));
  let t = viewPos4.xyz;

  // Discard if at or behind near plane
  let near = uniforms.params.y;
  if t.z > -near { return degen(); }

  // Near-depth fade: fade out splats within 3\xD7 the near distance so that
  // close-up splats don't cover the entire screen (matches Blender's behaviour).
  let depth     = -t.z;
  let nearFade  = clamp((depth - near) / (near * 2.0), 0.0, 1.0);

  // \u2500\u2500 3-D covariance \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let splatScale = uniforms.params.x;
  let s = g.scale * splatScale;
  let R = quatToMat3(g.quat);
  // M = R * diag(s); Cov3D = M * M\u1D40
  let M    = mat3x3<f32>(R[0]*s.x, R[1]*s.y, R[2]*s.z);
  let cov3 = M * transpose(M);

  // \u2500\u2500 Project covariance to 2-D \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  // Perspective Jacobian at view-space point t (camera looks down \u2013Z)
  let tz2 = t.z * t.z;
  let fx  = uniforms.focal.x;
  let fy  = uniforms.focal.y;
  let J = mat3x3<f32>(
    vec3<f32>(fx / (-t.z),  0.0,           0.0),
    vec3<f32>(0.0,          fy / (-t.z),   0.0),
    vec3<f32>(fx*t.x / tz2, fy*t.y / tz2, 0.0)
  );
  // Combined rotation: view \xD7 part \u2014 maps splat-local covariance to view space.
  // For single-part (identity partMat) this reduces to just the view rotation.
  let R_part = mat3x3<f32>(partMat[0].xyz, partMat[1].xyz, partMat[2].xyz);
  let W_view = mat3x3<f32>(uniforms.view[0].xyz, uniforms.view[1].xyz, uniforms.view[2].xyz);
  let W      = W_view * R_part;
  let T      = J * W;
  let cov2   = T * cov3 * transpose(T);

  // Extract 2\xD72 + low-pass filter (anti-aliasing)
  let a = cov2[0][0] + 0.3;   // \u03A3_xx  (cov2[col][row])
  let b = cov2[0][1];          // \u03A3_xy
  let c = cov2[1][1] + 0.3;   // \u03A3_yy

  // \u2500\u2500 Inverse covariance (conic) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let det = a*c - b*b;
  if det < 1e-4 { return degen(); }
  let inv = 1.0 / det;
  // conic = (A, B, C) such that Mahalanobis\xB2 = A\xB7dx\xB2 + 2B\xB7dx\xB7dy + C\xB7dy\xB2
  let conic = vec3<f32>(c*inv, -b*inv, a*inv);

  // \u2500\u2500 Bounding radius (3\u03C3 of largest eigenvalue) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let mid      = 0.5 * (a + c);
  let disc     = sqrt(max(0.1, mid*mid - det));
  let rawRadius = ceil(3.0 * sqrt(mid + disc));

  // Clamp screen-space radius: splats larger than half the viewport height
  // are faded out and capped, preventing nearby splats from covering the screen.
  let maxRadius = uniforms.viewport.y * 0.5;
  let sizeFade  = clamp(maxRadius / max(rawRadius, 1.0), 0.0, 1.0);
  let radius    = min(rawRadius, maxRadius);

  // \u2500\u2500 Screen-space quad placement \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  let clip    = uniforms.proj * viewPos4;
  let ndcXY   = clip.xy / clip.w;
  let pixOff  = corner * radius;
  let ndcOff  = pixOff / (uniforms.viewport * 0.5);

  var o: VOut;
  o.clipPos = vec4<f32>(ndcXY + ndcOff, clip.z / clip.w, 1.0);
  o.color   = vec4<f32>(g.color.rgb, g.color.a * nearFade * sizeFade);
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
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;var Wt=0,Vt=16,vt=32,wt=34,it=36,bt=40,Q=class{constructor(t,e){this.canvas=t,this.background=St(e),this.device=null,this.context=null,this.pipeline=null,this.bindGroup=null,this._uniformBuf=null,this._gaussianBuf=null,this._orderBuf=null,this._transformBuf=null,this._uniforms=new Float32Array(bt),this._uniforms[it]=1,this._numSplats=0}async init(){if(!navigator.gpu)throw new Error("WebGPU is not supported in this browser.");let t=await navigator.gpu.requestAdapter();if(!t)throw new Error("No WebGPU adapter found.");this.device=await t.requestDevice({requiredLimits:{maxStorageBufferBindingSize:t.limits.maxStorageBufferBindingSize,maxBufferSize:t.limits.maxBufferSize}}),this.context=this.canvas.getContext("webgpu"),this._format=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this._format,alphaMode:"premultiplied"}),this._createPipeline(),this._uniformBuf=this._createBuffer(bt*4,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);let e=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);this._transformBuf=this._createBuffer(64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.device.queue.writeBuffer(this._transformBuf,0,e)}uploadGaussians(t,e){this._numSplats=e,this._gaussianBuf&&this._gaussianBuf.destroy(),this._gaussianBuf=this.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this._gaussianBuf,0,t),this._orderBuf&&this._orderBuf.destroy(),this._orderBuf=this.device.createBuffer({size:e*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this._rebuildBindGroup()}updateUniforms({view:t,proj:e,width:s,height:n,focal:r,near:o=.1}){let a=this._uniforms;a.set(t,Wt),a.set(e,Vt),a[vt]=s,a[vt+1]=n,a[wt]=r,a[wt+1]=r,a[it+1]=o,this.device.queue.writeBuffer(this._uniformBuf,0,a)}updateOrder(t,e){this.device.queue.writeBuffer(this._orderBuf,0,t.buffer,0,e*4)}uploadTransforms(t){let e=new Float32Array(t.length*16);for(let s=0;s<t.length;s++)e.set(t[s],s*16);this._transformBuf&&this._transformBuf.destroy(),this._transformBuf=this._createBuffer(e.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST),this.device.queue.writeBuffer(this._transformBuf,0,e),this._gaussianBuf&&this._rebuildBindGroup()}updateTransforms(t){this._transformBuf&&this.device.queue.writeBuffer(this._transformBuf,0,t)}patchGaussians(t,e){this._gaussianBuf&&this.device.queue.writeBuffer(this._gaussianBuf,e*64,t)}setSplatScale(t){this._uniforms[it]=t}setBackground(t){this.background=St(t)}draw(){if(!this._numSplats||!this.bindGroup)return;let t=this.device.createCommandEncoder(),e=t.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:this.background,loadOp:"clear",storeOp:"store"}]});e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this._numSplats,0,0),e.end(),this.device.queue.submit([t.finish()])}destroy(){this._uniformBuf?.destroy(),this._gaussianBuf?.destroy(),this._orderBuf?.destroy(),this._transformBuf?.destroy(),this.context?.unconfigure()}_createPipeline(){let t=this.device.createShaderModule({code:xt});this.pipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this._format,blend:{color:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},primitive:{topology:"triangle-list",cullMode:"none"}})}_createBuffer(t,e){return this.device.createBuffer({size:t,usage:e})}_rebuildBindGroup(){this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._uniformBuf}},{binding:1,resource:{buffer:this._gaussianBuf}},{binding:2,resource:{buffer:this._orderBuf}},{binding:3,resource:{buffer:this._transformBuf}}]})}};function St(i){if(!i||i==="transparent")return{r:0,g:0,b:0,a:0};if(Array.isArray(i))return{r:i[0],g:i[1],b:i[2],a:i[3]??1};if(typeof i=="string"){let t=i.replace("#","");if(t.length===6)return{r:parseInt(t.slice(0,2),16)/255,g:parseInt(t.slice(2,4),16)/255,b:parseInt(t.slice(4,6),16)/255,a:1};if(t.length===8)return{r:parseInt(t.slice(0,2),16)/255,g:parseInt(t.slice(2,4),16)/255,b:parseInt(t.slice(4,6),16)/255,a:parseInt(t.slice(6,8),16)/255}}return{r:0,g:0,b:0,a:1}}var J=class{constructor({fov:t=60,near:e=.1,far:s=2e3}={}){this.fov=t*Math.PI/180,this.near=e,this.far=s,this.theta=0,this.phi=.2,this.radius=5,this.target=[0,0,0],this.enabled=!0,this.panEnabled=!0,this.thetaMin=null,this.thetaMax=null,this.phiMin=null,this.phiMax=null,this._drag=null,this._touches=[],this.viewMatrix=new Float32Array(16),this.projMatrix=new Float32Array(16)}attach(t){this._canvas=t,this._onMouseDown=e=>this._mouseDown(e),this._onMouseMove=e=>this._mouseMove(e),this._onMouseUp=()=>{this._drag=null},this._onWheel=e=>this._wheel(e),this._onTouchStart=e=>this._touchStart(e),this._onTouchMove=e=>this._touchMove(e),this._onTouchEnd=e=>this._touchEnd(e),this._onCtxMenu=e=>e.preventDefault(),t.addEventListener("mousedown",this._onMouseDown),t.addEventListener("mousemove",this._onMouseMove),t.addEventListener("mouseup",this._onMouseUp),t.addEventListener("mouseleave",this._onMouseUp),t.addEventListener("wheel",this._onWheel,{passive:!1}),t.addEventListener("touchstart",this._onTouchStart,{passive:!1}),t.addEventListener("touchmove",this._onTouchMove,{passive:!1}),t.addEventListener("touchend",this._onTouchEnd),t.addEventListener("contextmenu",this._onCtxMenu)}detach(){let t=this._canvas;t&&(t.removeEventListener("mousedown",this._onMouseDown),t.removeEventListener("mousemove",this._onMouseMove),t.removeEventListener("mouseup",this._onMouseUp),t.removeEventListener("mouseleave",this._onMouseUp),t.removeEventListener("wheel",this._onWheel),t.removeEventListener("touchstart",this._onTouchStart),t.removeEventListener("touchmove",this._onTouchMove),t.removeEventListener("touchend",this._onTouchEnd),t.removeEventListener("contextmenu",this._onCtxMenu),this._canvas=null)}_mouseDown(t){this.enabled&&(t.button===2&&!this.panEnabled||(this._drag={x:t.clientX,y:t.clientY,button:t.button},t.preventDefault()))}_mouseMove(t){if(!this._drag)return;let e=t.clientX-this._drag.x,s=t.clientY-this._drag.y;this._drag.x=t.clientX,this._drag.y=t.clientY,this._drag.button===2?this._pan(e,s):this._orbit(e,s)}_wheel(t){if(!this.enabled)return;t.preventDefault();let e=t.deltaY>0?1.1:.9;this.radius=Math.max(.01,this.radius*e)}_touchStart(t){this.enabled&&(t.preventDefault(),this._touches=Array.from(t.touches).map(e=>({id:e.identifier,x:e.clientX,y:e.clientY})))}_touchMove(t){if(!this.enabled)return;t.preventDefault();let e=this._touches,s=Array.from(t.touches).map(n=>({id:n.identifier,x:n.clientX,y:n.clientY}));if(s.length===1&&e.length===1){let n=s[0].x-e[0].x,r=s[0].y-e[0].y;this._orbit(n,r)}else if(s.length===2&&e.length===2){let n=Math.hypot(e[1].x-e[0].x,e[1].y-e[0].y),r=Math.hypot(s[1].x-s[0].x,s[1].y-s[0].y);if(n>0&&(this.radius=Math.max(.01,this.radius*(n/r))),this.panEnabled){let o=(e[0].x+e[1].x)*.5,a=(e[0].y+e[1].y)*.5,c=(s[0].x+s[1].x)*.5,f=(s[0].y+s[1].y)*.5;this._pan(c-o,f-a)}}this._touches=s}_touchEnd(t){this._touches=Array.from(t.touches).map(e=>({id:e.identifier,x:e.clientX,y:e.clientY}))}_orbit(t,e){this.theta-=t*.005,this.phi=Math.max(-Math.PI/2+.01,Math.min(Math.PI/2-.01,this.phi+e*.005)),this.thetaMin!==null&&(this.theta=Math.max(this.thetaMin,this.theta)),this.thetaMax!==null&&(this.theta=Math.min(this.thetaMax,this.theta)),this.phiMin!==null&&(this.phi=Math.max(this.phiMin,this.phi)),this.phiMax!==null&&(this.phi=Math.min(this.phiMax,this.phi))}constrainAngles(t,e){if(t!==null){let s=t*Math.PI/180;this.thetaMin=this.theta-s,this.thetaMax=this.theta+s}else this.thetaMin=null,this.thetaMax=null;if(e!==null){let s=e*Math.PI/180;this.phiMin=Math.max(-Math.PI/2+.01,this.phi-s),this.phiMax=Math.min(Math.PI/2-.01,this.phi+s)}else this.phiMin=null,this.phiMax=null}clearConstraints(){this.thetaMin=null,this.thetaMax=null,this.phiMin=null,this.phiMax=null}_pan(t,e){let s=this.radius*.001,n=this._cameraRight(),r=this._cameraUp();this.target[0]-=(n[0]*t-r[0]*e)*s,this.target[1]-=(n[1]*t-r[1]*e)*s,this.target[2]-=(n[2]*t-r[2]*e)*s}_cameraRight(){return[this.viewMatrix[0],this.viewMatrix[4],this.viewMatrix[8]]}_cameraUp(){return[this.viewMatrix[1],this.viewMatrix[5],this.viewMatrix[9]]}update(t,e){let s=this._eye();Xt(s,this.target,[0,1,0],this.viewMatrix),Zt(this.fov,t/e,this.near,this.far,this.projMatrix)}_eye(){let t=Math.cos(this.phi),e=Math.sin(this.phi),s=Math.cos(this.theta),n=Math.sin(this.theta);return[this.target[0]+this.radius*t*n,this.target[1]+this.radius*e,this.target[2]+this.radius*t*s]}focalLength(t){return t*.5/Math.tan(this.fov*.5)}setFromLookAt(t,e){this.target=[e[0],e[1],e[2]];let s=t[0]-e[0],n=t[1]-e[1],r=t[2]-e[2];this.radius=Math.hypot(s,n,r)||.001,this.phi=Math.asin(Math.max(-1,Math.min(1,n/this.radius))),this.theta=Math.atan2(s,r)}fitScene(t,e){this._sceneBounds(t,e),this.theta=0,this.phi=.2}focusScene(t,e){this._sceneBounds(t,e)}_sceneBounds(t,e){let s=1/0,n=1/0,r=1/0,o=-1/0,a=-1/0,c=-1/0;for(let m=0;m<e;m++){let l=m*16,u=t[l],h=t[l+1],d=t[l+2];u<s&&(s=u),u>o&&(o=u),h<n&&(n=h),h>a&&(a=h),d<r&&(r=d),d>c&&(c=d)}this.target=[(s+o)*.5,(n+a)*.5,(r+c)*.5];let f=Math.max(o-s,a-n,c-r)*.5;this.radius=f/Math.tan(this.fov*.5)*1.2}};function Xt(i,t,e,s){let[n,r,o]=i,[a,c,f]=t,[m,l,u]=e,h=n-a,d=r-c,p=o-f,_=Math.hypot(h,d,p);h/=_,d/=_,p/=_;let v=l*p-u*d,M=u*h-m*p,F=m*d-l*h,b=Math.hypot(v,M,F);v/=b,M/=b,F/=b;let P=d*F-p*M,U=p*v-h*F,y=h*M-d*v;s[0]=v,s[1]=P,s[2]=h,s[3]=0,s[4]=M,s[5]=U,s[6]=d,s[7]=0,s[8]=F,s[9]=y,s[10]=p,s[11]=0,s[12]=-(v*n+M*r+F*o),s[13]=-(P*n+U*r+y*o),s[14]=-(h*n+d*r+p*o),s[15]=1}function Zt(i,t,e,s,n){let r=1/Math.tan(i*.5),o=e-s;n[0]=r/t,n[1]=0,n[2]=0,n[3]=0,n[4]=0,n[5]=r,n[6]=0,n[7]=0,n[8]=0,n[9]=0,n[10]=s/o,n[11]=-1,n[12]=0,n[13]=0,n[14]=e*s/o,n[15]=0}var Qt=`
function radixSort32(keys, idx, tmp_k, tmp_i, counts, pfx, N) {
  for (let pass = 0; pass < 4; pass++) {
    const shift = pass * 8;
    counts.fill(0);
    for (let i = 0; i < N; i++) counts[(keys[i] >>> shift) & 0xff]++;
    pfx[0] = 0;
    for (let i = 1; i < 256; i++) pfx[i] = pfx[i-1] + counts[i-1];
    for (let i = 0; i < N; i++) {
      const b = (keys[i] >>> shift) & 0xff;
      tmp_k[pfx[b]]   = keys[i];
      tmp_i[pfx[b]++] = idx[i];
    }
    // swap
    const k = keys; keys = tmp_k; tmp_k = k;
    const x = idx;  idx  = tmp_i; tmp_i = x;
  }
  // after 4 passes (even) idx holds the result
  return idx;
}
`,Jt=Qt+`
const u32 = new Uint32Array(1);
const f32 = new Float32Array(u32.buffer);

let keys0, keys1, idx0, idx1, counts, pfx, allocN = 0;
function alloc(N) {
  if (N <= allocN) return;
  allocN = N;
  keys0  = new Uint32Array(N);
  keys1  = new Uint32Array(N);
  idx0   = new Uint32Array(N);
  idx1   = new Uint32Array(N);
  counts = new Uint32Array(256);
  pfx    = new Uint32Array(256);
}

self.onmessage = function(e) {
  const depths = e.data.depths;
  const N      = e.data.N;
  alloc(N);

  // Convert negative floats \u2192 sortable uint32 by flipping all bits
  const dv = new DataView(depths.buffer);
  for (let i = 0; i < N; i++) {
    keys0[i] = dv.getUint32(i * 4, true) ^ 0xffffffff;
    idx0[i]  = i;
  }

  const sorted = radixSort32(keys0, idx0, keys1, idx1, counts, pfx, N);
  const result = sorted.slice(0, N);
  self.postMessage({ order: result }, [result.buffer]);
};
`;function Mt(i){let t=new Uint32Array(i),e=new Uint32Array(i),s=new Uint32Array(i),n=new Uint32Array(i),r=new Uint32Array(256),o=new Uint32Array(256);function a(c,f){let m=new DataView(c.buffer);for(let l=0;l<f;l++)t[l]=m.getUint32(l*4,!0)^4294967295,s[l]=l;for(let l=0;l<4;l++){let u=l*8;r.fill(0);for(let h=0;h<f;h++)r[t[h]>>>u&255]++;o[0]=0;for(let h=1;h<256;h++)o[h]=o[h-1]+r[h-1];for(let h=0;h<f;h++){let d=t[h]>>>u&255;e[o[d]]=t[h],n[o[d]++]=s[h]}[t,e]=[e,t],[s,n]=[n,s]}return s}return a}function W(i){if(typeof Worker>"u")return Mt(i);let t;try{let r=new Blob([Jt],{type:"application/javascript"}),o=URL.createObjectURL(r);t=new Worker(o),URL.revokeObjectURL(o)}catch{return Mt(i)}let e=new Uint32Array(i);for(let r=0;r<i;r++)e[r]=r;let s=!1,n=null;return t.onmessage=r=>{if(e=r.data.order,s=!1,n){let{depths:o,N:a}=n;n=null,s=!0,t.postMessage({depths:o,N:a},[o.buffer])}},function(o,a){let c=new Float32Array(a);return c.set(o.subarray(0,a)),s?n={depths:c,N:a}:(s=!0,t.postMessage({depths:c,N:a},[c.buffer])),e}}async function N(i,t){let e=await fetch(i);if(!e.ok)throw new Error(`HTTP ${e.status} loading ${i}`);let s=parseInt(e.headers.get("content-length")||"0",10),n=e.body.getReader(),r=[],o=0;for(;;){let{done:f,value:m}=await n.read();if(f)break;r.push(m),o+=m.byteLength,t&&s>0&&t(o/s)}let a=new Uint8Array(o),c=0;for(let f of r)a.set(f,c),c+=f.byteLength;return a.buffer}async function At(i,t){let e=await N(i,t);return Pt(e)}function Pt(i){let e=Math.floor(i.byteLength/32);if(e===0)throw new Error("Empty or invalid .splat file");let s=new DataView(i),n=new Float32Array(e*16);for(let r=0;r<e;r++){let o=r*32,a=r*16;n[a+0]=s.getFloat32(o+0,!0),n[a+1]=s.getFloat32(o+4,!0),n[a+2]=s.getFloat32(o+8,!0),n[a+4]=s.getUint8(o+24)/255,n[a+5]=s.getUint8(o+25)/255,n[a+6]=s.getUint8(o+26)/255,n[a+7]=s.getUint8(o+27)/255,n[a+8]=s.getFloat32(o+12,!0),n[a+9]=s.getFloat32(o+16,!0),n[a+10]=s.getFloat32(o+20,!0);let c=(s.getUint8(o+28)-128)/128,f=(s.getUint8(o+29)-128)/128,m=(s.getUint8(o+30)-128)/128,l=(s.getUint8(o+31)-128)/128,u=Math.hypot(f,m,l,c)||1;n[a+12]=f/u,n[a+13]=m/u,n[a+14]=l/u,n[a+15]=c/u}return{data:n,count:e}}var nt=.28209479177387814;async function Ft(i,t){let e=await N(i,t);return Ut(e)}function Ut(i){let t=new Uint8Array(i),e=Ct(t);if(e<0)throw new Error("Invalid PLY: end_header not found");let{numVertices:s,propMap:n,stride:r,hasColor:o}=kt(new TextDecoder().decode(t.subarray(0,e)));if(s===0)throw new Error("PLY file contains no vertices");let a=new DataView(i,e);return{data:Et(a,n,r,o,s),count:s}}async function at(i,t){let e=await fetch(i);if(!e.ok)throw new Error(`HTTP ${e.status} loading ${i}`);let s=parseInt(e.headers.get("content-length")||"0",10),n=e.body.getReader(),r=new Uint8Array(0),o=0;for(;;){let{done:a,value:c}=await n.read();if(a)throw new Error("PLY stream ended before end_header");o+=c.byteLength,t&&s>0&&t(o/s);let f=new Uint8Array(r.length+c.length);f.set(r),f.set(c,r.length),r=f;let m=Ct(r);if(m>=0){let l=kt(new TextDecoder().decode(r.subarray(0,m))),u=r.slice(m),{numVertices:h,propMap:d,stride:p,hasColor:_}=l;return{numVertices:h,consume:async(M,F)=>{let b=u;for(;;){let{done:P,value:U}=await n.read();if(!P){o+=U.byteLength,F&&s>0&&F(o/s);let x=new Uint8Array(b.length+U.length);x.set(b),x.set(U,b.length),b=x}let y=Math.floor(b.length/p);if(y>0){let x=y*p;M(Et(new DataView(b.buffer,b.byteOffset,x),d,p,_,y),y),b=b.slice(x)}if(P)break}}}}if(r.length>65536)throw new Error("PLY header exceeds 64 KB")}}function Et(i,t,e,s,n){let r=new Float32Array(n*16);for(let o=0;o<n;o++){let a=o*e,c=o*16;r[c+0]=z(i,a,t.x),r[c+1]=z(i,a,t.y),r[c+2]=z(i,a,t.z),s?(r[c+4]=rt(.5+nt*z(i,a,t.f_dc_0)),r[c+5]=rt(.5+nt*z(i,a,t.f_dc_1)),r[c+6]=rt(.5+nt*z(i,a,t.f_dc_2))):(r[c+4]=1,r[c+5]=1,r[c+6]=1),r[c+7]=te(z(i,a,t.opacity)),r[c+8]=Math.exp(z(i,a,t.scale_0)),r[c+9]=Math.exp(z(i,a,t.scale_1)),r[c+10]=Math.exp(z(i,a,t.scale_2));let f=z(i,a,t.rot_0),m=z(i,a,t.rot_1),l=z(i,a,t.rot_2),u=z(i,a,t.rot_3),h=Math.hypot(m,l,u,f)||1;r[c+12]=m/h,r[c+13]=l/h,r[c+14]=u/h,r[c+15]=f/h}return r}function Ct(i){let t=[101,110,100,95,104,101,97,100,101,114];t:for(let e=0;e<=i.length-t.length;e++){for(let n=0;n<t.length;n++)if(i[e+n]!==t[n])continue t;let s=e+t.length;return i[s]===13&&s++,i[s]===10&&s++,s}return-1}function kt(i){let t=i.split(`
`),e=0,s=!1,n=[];for(let c of t){let f=c.trim().split(/\s+/);f[0]==="element"?(s=f[1]==="vertex",s&&(e=parseInt(f[2],10))):f[0]==="property"&&s&&n.push({type:f[1],name:f[2]})}let r={},o=0;for(let c of n)r[c.name]={offset:o,type:c.type},o+=Kt(c.type);let a=["x","y","z","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","opacity"];for(let c of a)if(!r[c])throw new Error(`PLY missing required property: ${c}`);return{numVertices:e,propMap:r,stride:o,hasColor:!!(r.f_dc_0&&r.f_dc_1&&r.f_dc_2)}}function Kt(i){switch(i){case"float":case"float32":case"int":case"uint":return 4;case"double":case"int64":case"uint64":return 8;case"short":case"ushort":case"int16":case"uint16":return 2;case"char":case"uchar":case"int8":case"uint8":return 1;default:return 4}}function z(i,t,e){let s=t+e.offset;switch(e.type){case"float":case"float32":return i.getFloat32(s,!0);case"double":return i.getFloat64(s,!0);case"int":case"int32":return i.getInt32(s,!0);case"uint":case"uint32":return i.getUint32(s,!0);case"short":case"int16":return i.getInt16(s,!0);case"ushort":case"uint16":return i.getUint16(s,!0);case"char":case"int8":return i.getInt8(s);case"uchar":case"uint8":return i.getUint8(s);default:return i.getFloat32(s,!0)}}function te(i){return 1/(1+Math.exp(-i))}function rt(i){return i<0?0:i>1?1:i}var zt=1347635022,ee=1/Math.SQRT2;async function Bt(i,t){let e=await N(i,t),s=await ie(e);return se(s)}function se(i){let t=new DataView(i),e=t.getUint32(0,!0),s=t.getUint32(4,!0),n=t.getUint32(8,!0),r=t.getUint8(12),o=t.getUint8(13);if(e!==zt)throw new Error(`Invalid .spz magic: 0x${e.toString(16).toUpperCase()} (expected 0x${zt.toString(16).toUpperCase()})`);(s<2||s>4)&&console.warn(`HoloSplat: .spz version ${s} is untested; attempting load anyway`);let a=1<<o,c=s>=3?4:3,f=16,m=f+n*9,l=m+n*1,u=l+n*3,h=u+n*3,d=new Float32Array(n*16);for(let p=0;p<n;p++){let _=p*16,v=f+p*9;d[_+0]=ot(t,v+0)/a,d[_+1]=ot(t,v+3)/a,d[_+2]=ot(t,v+6)/a;let M=l+p*3;d[_+4]=t.getUint8(M+0)/255,d[_+5]=t.getUint8(M+1)/255,d[_+6]=t.getUint8(M+2)/255,d[_+7]=t.getUint8(m+p)/255;let F=u+p*3;d[_+8]=Math.exp((t.getUint8(F+0)-128)/16),d[_+9]=Math.exp((t.getUint8(F+1)-128)/16),d[_+10]=Math.exp((t.getUint8(F+2)-128)/16);let b=h+p*c,P,U,y,x;if(s>=3){let T=t.getUint32(b,!0),G=T&3,R=ee/512,S=ct(T>>2&1023)*R,C=ct(T>>12&1023)*R,B=ct(T>>22&1023)*R,L=Math.sqrt(Math.max(0,1-S*S-C*C-B*B));switch(G){case 0:P=L,U=S,y=C,x=B;break;case 1:P=S,U=L,y=C,x=B;break;case 2:P=S,U=C,y=L,x=B;break;default:P=S,U=C,y=B,x=L}}else P=t.getInt8(b+0)/128,U=t.getInt8(b+1)/128,y=t.getInt8(b+2)/128,x=Math.sqrt(Math.max(0,1-P*P-U*U-y*y));let w=Math.hypot(P,U,y,x)||1;d[_+12]=P/w,d[_+13]=U/w,d[_+14]=y/w,d[_+15]=x/w}return{data:d,count:n}}function ot(i,t){let e=i.getUint8(t),s=i.getUint8(t+1),n=i.getInt8(t+2);return e|s<<8|n<<16}function ct(i){return i&512?i|4294966272:i}async function ie(i){if(typeof DecompressionStream>"u")throw new Error("DecompressionStream is not available in this environment");let t=new DecompressionStream("gzip"),e=t.writable.getWriter();return e.write(i),e.close(),new Response(t.readable).arrayBuffer()}function lt(i){let t=i.split(".");for(let e=t.length-1;e>=0;e--)if(!/^\d+$/.test(t[e]))return t[e];return t[0]}var K=class{constructor(t){if(!t.frames||t.frames.length===0)throw new Error("HoloSplat Animation: no frame data");this.fps=t.fps??24,this.frameCount=t.frameCount??Math.floor(t.frames.length/6),this.fov=t.fov??null,this.near=t.near??null,this.far=t.far??null,this.callouts=t.callouts??[],this.focalPoint=t.focalPoint??null,this.loop=!0,Array.isArray(t.markers)?this.markers=Object.fromEntries(t.markers.map(e=>[e.name,e.frame])):this.markers=t.markers??{},this._frames=new Float32Array(t.frames),this._objects=(t.objects??[]).map(e=>({id:e.id,frames:new Float32Array(e.frames)})),this._time=0,this._playing=!0}get duration(){return this.frameCount/this.fps}get time(){return this._time}get playing(){return this._playing}get objects(){return this._objects}play(){this._playing=!0}pause(){this._playing=!1}seek(t){this._time=Math.max(0,Math.min(this.duration,t))}seekFrame(t){this._time=Math.max(0,Math.min(this.frameCount-1,t))/this.fps}get frame(){return this._time*this.fps}tick(t){this._playing&&(this._time+=t,this._time>=this.duration&&(this._time=this.loop?this._time%this.duration:this.duration,this.loop||(this._playing=!1)))}getObjectFrames(){let t=Math.min(Math.floor(this._time*this.fps),this.frameCount-1);return this._objects.map(e=>{let s=t*7,n=e.frames;return{id:e.id,pos:[n[s],n[s+1],n[s+2]],quat:[n[s+3],n[s+4],n[s+5],n[s+6]]}})}getCameraFrame(){let e=Math.min(Math.floor(this._time*this.fps),this.frameCount-1)*6,s=this._frames;return{eye:[s[e],s[e+1],s[e+2]],target:[s[e]+s[e+3],s[e+1]+s[e+4],s[e+2]+s[e+5]]}}};async function ht(i){let t=await fetch(i);if(!t.ok)throw new Error(`HoloSplat: failed to load animation "${i}" (HTTP ${t.status})`);return new K(await t.json())}var $=class{constructor(t={}){let{canvas:e,background:s="#000000",fov:n=60,near:r=.1,far:o=2e3,splatScale:a=1,autoRotate:c=!1,flipY:f=!1,onProgress:m,onError:l}=t;this._canvas=ae(e),this._onProgress=m,this._onError=l,this._autoRotate=c,this._splatScale=a,this._flipY=f,this._renderer=new Q(this._canvas,s),this._camera=new J({fov:n,near:r,far:o}),this._gaussians=null,this._numSplats=0,this._depths=null,this._sort=null,this._rafId=null,this._running=!1,this._resizeObs=null,this._partIndex={},this._partTransforms=[j.slice()],this._partTransFlat=j.slice(),this._animation=null,this._animPaused=!1,this._cameraFree=!1,this._sceneReady=!1,this._lastTick=null,this.onFrame=null}async init(){await this._renderer.init(),this._renderer.setSplatScale(this._splatScale),this._camera.attach(this._canvas),this._observeResize(),this._updateSize()}async load(t){if(this._sceneReady=!1,t.split("?")[0].split(".").pop().toLowerCase()==="ply")try{await this._loadPlyStreamSingle(t);return}catch(r){if(!/^HTTP 4/.test(r.message))throw r}let{data:s,count:n}=await Tt(t,r=>this._onProgress?.(r));this._flipY&&V(s,n);for(let r=0;r<n;r++)s[r*16+3]=0;if(this._gaussians=s,this._numSplats=n,this._depths=new Float32Array(n),this._sort=W(n),this._partIndex={},this._partTransforms=[j.slice()],this._partTransFlat=j.slice(),this._renderer.uploadGaussians(s,n),this._renderer.uploadTransforms(this._partTransforms),this._camera.fitScene(s,n),this._animation){let{eye:r,target:o}=this._animation.getCameraFrame();this._camera.setFromLookAt(r,this._animation.focalPoint??o)}this._sceneReady=!0}async _loadPlyStreamSingle(t){let{numVertices:e,consume:s}=await at(t,o=>this._onProgress?.(o*.02)),n=new Float32Array(e*16);if(this._gaussians=n,this._numSplats=e,this._depths=new Float32Array(e),this._sort=W(e),this._partIndex={},this._partTransforms=[j.slice()],this._partTransFlat=j.slice(),this._renderer.uploadGaussians(n,e),this._renderer.uploadTransforms(this._partTransforms),this._animation){let{eye:o,target:a}=this._animation.getCameraFrame();this._camera.setFromLookAt(o,this._animation.focalPoint??a)}let r=0;await s((o,a)=>{for(let c=0;c<a;c++)o[c*16+3]=0;this._flipY&&V(o,a),this._renderer.patchGaussians(o,r),this._gaussians.set(o,r*16),r+=a},o=>this._onProgress?.(o)),this._animation||this._camera.fitScene(this._gaussians,this._numSplats),this._sceneReady=!0}async loadParts(t){let e=Object.keys(t);if(e.length===0)throw new Error("HoloSplat: loadParts called with empty map");this._sceneReady=!1;let s=new Array(e.length).fill(0),n=()=>{this._onProgress&&this._onProgress(s.reduce((l,u)=>l+u,0)/e.length)},r=await Promise.all(e.map(async(l,u)=>{let h=t[l];if(h.split("?")[0].split(".").pop().toLowerCase()==="ply")try{let{numVertices:v,consume:M}=await at(h,F=>{s[u]=F*.03,n()});return{id:l,idx:u,kind:"stream",numVertices:v,consume:M}}catch(v){if(!/^HTTP 4/.test(v.message))throw v}let{data:p,count:_}=await Tt(h,v=>{s[u]=v*.9,n()});return s[u]=.9,{id:l,idx:u,kind:"loaded",data:p,count:_}})),o=r.map(l=>l.kind==="stream"?l.numVertices:l.count),a=o.reduce((l,u)=>l+u,0),c=[];{let l=0;for(let u of o)c.push(l),l+=u}let f=new Float32Array(a*16);for(let l of r){if(l.kind!=="loaded")continue;let{data:u,count:h,idx:d}=l;for(let p=0;p<h;p++)u[p*16+3]=d;this._flipY&&V(u,h),f.set(u,c[d]*16),s[d]=1}this._partIndex={},e.forEach((l,u)=>{this._partIndex[l]=u}),this._partTransforms=o.map(()=>j.slice()),this._partTransFlat=new Float32Array(o.length*16);for(let l=0;l<o.length;l++)this._partTransFlat.set(j,l*16);if(this._gaussians=f,this._numSplats=a,this._depths=new Float32Array(a),this._sort=W(a),this._renderer.uploadGaussians(f,a),this._renderer.uploadTransforms(this._partTransforms),this._camera.fitScene(f,a),this._animation){let{eye:l,target:u}=this._animation.getCameraFrame();this._camera.setFromLookAt(l,this._animation.focalPoint??u)}n();let m=r.filter(l=>l.kind==="stream");if(m.length===0){this._sceneReady=!0;return}await Promise.all(m.map(async l=>{let{idx:u,consume:h}=l,d=c[u];await h((p,_)=>{for(let v=0;v<_;v++)p[v*16+3]=u;this._flipY&&V(p,_),this._renderer.patchGaussians(p,d),this._gaussians.set(p,d*16),d+=_},p=>{s[u]=.03+.97*p,n()}),s[u]=1,n()})),this._animation||this._camera.fitScene(this._gaussians,this._numSplats),this._sceneReady=!0}start(){this._running||(this._running=!0,this._tick())}stop(){this._running=!1,this._rafId&&cancelAnimationFrame(this._rafId),this._rafId=null}destroy(){this.stop(),this._camera.detach(),this._renderer.destroy(),this._resizeObs?.disconnect()}setBackground(t){this._renderer.setBackground(t)}setSplatScale(t){this._splatScale=t,this._renderer.setSplatScale(t)}setAutoRotate(t){this._autoRotate=t}setFlipY(t){!!t!==this._flipY&&(this._flipY=!!t,this._gaussians&&(V(this._gaussians,this._numSplats),this._renderer.uploadGaussians(this._gaussians,this._numSplats),this._camera.fitScene(this._gaussians,this._numSplats)))}setAnimationPaused(t){this._animPaused=t}setCameraFree(t){this._cameraFree=!!t}_syncCameraMode(){let t=this._animation.markers,e=this._animation.frame,s=null,n=-1,r=!1;for(let a of Object.keys(t)){if(!a.startsWith("hs-"))continue;r=!0;let c=t[a];c<=e&&c>n&&(n=c,s=a)}if(!r)return;let o=s!==null&&s!=="hs-locked";if(o!==this._cameraFree)if(this._cameraFree=o,o){let{eye:a,target:c}=this._animation.getCameraFrame(),f=this._animation.focalPoint;this._camera.setFromLookAt(a,f??c),f&&(this._camera.panEnabled=!1);let m=s.match(/h(\d+)/),l=s.match(/v(\d+)/);this._camera.constrainAngles(m?parseInt(m[1],10):null,l?parseInt(l[1],10):null)}else this._animation.focalPoint&&(this._camera.panEnabled=!0),this._camera.clearConstraints()}resetCamera(){this._camera.fitScene(this._gaussians,this._numSplats)}focusCamera(){this._camera.focusScene(this._gaussians,this._numSplats)}setGaussians(t,e,s=!1){this._gaussians=t,this._numSplats=e,this._depths=new Float32Array(e),this._sort=W(e),this._sceneReady=!0,this._renderer.uploadGaussians(t,e),s&&this._camera.fitScene(t,e)}uploadDisplay(t){this._numSplats&&this._renderer.uploadGaussians(t,this._numSplats)}setAnimation(t){if(this._animation=t,!t)return;t.fov!=null&&(this._camera.fov=t.fov*Math.PI/180),t.near!=null&&(this._camera.near=t.near),t.far!=null&&(this._camera.far=t.far);let{eye:e,target:s}=t.getCameraFrame();this._camera.setFromLookAt(e,t.focalPoint??s)}async loadAnimationUrl(t){let e=await ht(t);return this.setAnimation(e),e}projectCallouts(t){let e=this._camera.viewMatrix,s=this._camera.projMatrix,n=this._canvas.clientWidth,r=this._canvas.clientHeight,o=[];for(let a of t){let[c,f,m]=a.pos,l=e[0]*c+e[4]*f+e[8]*m+e[12],u=e[1]*c+e[5]*f+e[9]*m+e[13],h=e[2]*c+e[6]*f+e[10]*m+e[14];if(h>=0){o.push({id:a.id,visible:!1,x:0,y:0});continue}let d=-h,p=(s[0]*l/d*.5+.5)*n,_=(1-(s[5]*u/d*.5+.5))*r;o.push({id:a.id,visible:!0,x:p,y:_})}return o}get camera(){return this._camera}_tick(){if(!this._running)return;this._rafId=requestAnimationFrame(()=>this._tick());let t=performance.now(),e=this._lastTick?Math.min((t-this._lastTick)/1e3,.1):0;this._lastTick=t;let s=this._canvas.width,n=this._canvas.height;if(this._animation){if(this._animPaused||(this._sceneReady&&this._animation.tick(e),this._syncCameraMode()),!this._cameraFree){let{eye:m,target:l}=this._animation.getCameraFrame();this._camera.setFromLookAt(m,l)}let f=this._animation.getObjectFrames();if(f.length>0){let m=!1;for(let{id:l,pos:u,quat:h}of f){let d=this._partIndex[l];d!==void 0&&(this._partTransFlat.set(re(u,h),d*16),m=!0)}m&&this._renderer.updateTransforms(this._partTransFlat)}}else this._autoRotate&&(this._camera.theta+=.005);if(!this._numSplats)return;this._camera.update(s,n),this.onFrame&&this.onFrame(this._camera.viewMatrix,this._camera.projMatrix,s,n);let r=this._camera.viewMatrix,o=this._camera.projMatrix,a=this._camera.focalLength(n);this._computeDepths(r);let c=this._sort(this._depths,this._numSplats);this._renderer.updateUniforms({view:r,proj:o,width:s,height:n,focal:a,near:this._camera.near}),this._renderer.updateOrder(c,this._numSplats),this._renderer.draw()}_computeDepths(t){let e=t[2],s=t[6],n=t[10],r=t[14],o=[],a=this._partTransFlat,c=this._partTransforms.length;for(let u=0;u<c;u++){let h=u*16;o.push([e*a[h]+s*a[h+1]+n*a[h+2],e*a[h+4]+s*a[h+5]+n*a[h+6],e*a[h+8]+s*a[h+9]+n*a[h+10],e*a[h+12]+s*a[h+13]+n*a[h+14]+r])}let f=this._gaussians,m=this._depths,l=this._numSplats;for(let u=0;u<l;u++){let h=u*16,d=o[f[h+3]];m[u]=d[0]*f[h]+d[1]*f[h+1]+d[2]*f[h+2]+d[3]}}_observeResize(){typeof ResizeObserver>"u"||(this._resizeObs=new ResizeObserver(()=>this._updateSize()),this._resizeObs.observe(this._canvas))}_updateSize(){let t=window.devicePixelRatio||1,e=Math.round(this._canvas.clientWidth*t),s=Math.round(this._canvas.clientHeight*t);e&&s&&(this._canvas.width!==e||this._canvas.height!==s)&&(this._canvas.width=e,this._canvas.height=s)}},j=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);function V(i,t){for(let e=0;e<t;e++){let s=e*16;i[s+1]=-i[s+1],i[s+2]=-i[s+2];let n=i[s+12],r=i[s+13],o=i[s+14],a=i[s+15];i[s+12]=a,i[s+13]=-o,i[s+14]=r,i[s+15]=-n}}var ft=["spz","ply","splat"],ne={ply:Ft,spz:Bt};async function Tt(i,t){let e=i.split("?")[0],s=e.lastIndexOf("."),n=s>=0?e.slice(s+1).toLowerCase():"",r=ft.includes(n),o=r?[n,...ft.filter(f=>f!==n)]:ft,a=r?i.slice(0,i.lastIndexOf(".")):i,c;for(let f of o){let m=`${a}.${f}`,l=ne[f]??At;try{return await l(m,t)}catch(u){if(!/^HTTP 4/.test(u.message))throw u;c=u}}throw new Error(`HoloSplat: splat file not found as .spz / .ply / .splat \u2014 "${a}"`)}function re(i,t){let[e,s,n,r]=t,[o,a,c]=i,f=e*2,m=s*2,l=n*2,u=e*f,h=e*m,d=e*l,p=s*m,_=s*l,v=n*l,M=r*f,F=r*m,b=r*l;return new Float32Array([1-p-v,h+b,d-F,0,h-b,1-u-v,_+M,0,d+F,_-M,1-u-p,0,o,a,c,1])}function ae(i){if(!i)throw new Error("HoloSplat: canvas option is required");if(typeof i=="string"){let t=document.querySelector(i);if(!t)throw new Error(`HoloSplat: canvas selector "${i}" not found`);return t}return i}var oe=`
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
.hs-lines{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;}
.hs-dot{fill:#3a7aff;stroke:#fff;stroke-width:2;}
.hs-line{stroke:rgba(255,255,255,.55);stroke-width:1.5;}
.hs-callout{position:absolute;pointer-events:auto;}
.hs-callout--hidden{display:none;}
`,Rt=!1;function ce(){if(Rt||typeof document>"u")return;Rt=!0;let i=document.createElement("style");i.textContent=oe,document.head.appendChild(i)}function qt(i,t={}){ce();let e=typeof i=="string"?document.querySelector(i):i;if(!e)throw new Error(`HoloSplat: container not found \u2014 "${i}"`);let s=e.getAttribute("data-hs-scene")||e.getAttribute("data-holosplat")||void 0,n=e.getAttribute("data-hs-animation")||e.getAttribute("data-holosplat-anim")||void 0,r=e.getAttribute("data-hs-flip-y")==="true",{scene:o,src:a,parts:c,animation:f=n,partsDir:m,partsExt:l="",background:u="transparent",fov:h=60,near:d=.1,far:p=2e3,splatScale:_=1,autoRotate:v=!1,flipY:M=r,onLoad:F,onProgress:b,onError:P}=t,U=o??a??s;e.classList.add("hs-player");let y=document.createElement("canvas"),x=document.createElement("div");x.className="hs-callouts";let w=document.createElement("div");w.className="hs-overlay",w.innerHTML='<div class="hs-spinner"></div><div class="hs-bar-wrap"><div class="hs-bar"></div></div><div class="hs-msg"></div>',e.appendChild(y),e.appendChild(x),e.appendChild(w);let T=w.querySelector(".hs-spinner"),G=w.querySelector(".hs-bar-wrap"),R=w.querySelector(".hs-bar"),S=w.querySelector(".hs-msg");function C(){T.style.display="",G.style.display="",S.textContent="",S.className="hs-msg",w.style.display="flex"}function B(){w.style.display="none"}function L(g){T.style.display="none",G.style.display="none",S.textContent=g,S.className="hs-msg hs-err",w.style.display="flex"}w.style.display="none";let E=new $({canvas:y,background:u,fov:h,near:d,far:p,splatScale:_,autoRotate:v,flipY:M,onProgress:g=>{R.style.width=`${(g*100).toFixed(0)}%`,b&&b(g)}}),H={},tt="http://www.w3.org/2000/svg";function Nt(g){x.innerHTML="";for(let k of Object.keys(H))delete H[k];if(!g.length)return;let A=document.createElementNS(tt,"svg");A.setAttribute("class","hs-lines"),x.appendChild(A);for(let k of g){let q=document.createElementNS(tt,"line");q.setAttribute("class","hs-line"),A.appendChild(q);let I=document.createElementNS(tt,"circle");I.setAttribute("class","hs-dot"),I.setAttribute("r","5"),A.appendChild(I);let O=e.querySelector(`.hs-callout[data-id="${k.id}"]`)??document.querySelector(`.hs-callout[data-id="${k.id}"]`);O||(O=document.createElement("div"),O.className="hs-callout",O.dataset.id=k.id),x.appendChild(O),H[k.id]={card:O,dot:I,line:q}}}E.onFrame=()=>{if(!E._animation?.callouts.length)return;let g=E.projectCallouts(E._animation.callouts);for(let{id:A,visible:k,x:q,y:I}of g){let O=H[A];if(!O)continue;let{card:D,dot:Z,line:Y}=O;if(k){let $t=parseFloat(D.dataset.offsetX??D.dataset.ox??80),Ht=parseFloat(D.dataset.offsetY??D.dataset.oy??-40),gt=q+$t,yt=I+Ht;Z.setAttribute("cx",q),Z.setAttribute("cy",I),Y.setAttribute("x1",q),Y.setAttribute("y1",I),Y.setAttribute("x2",gt),Y.setAttribute("y2",yt),Z.style.display="",Y.style.display="",D.style.left=gt+"px",D.style.top=yt+"px",D.classList.remove("hs-callout--hidden")}else Z.style.display="none",Y.style.display="none",D.classList.add("hs-callout--hidden")}};async function mt(g){C(),R.style.width="0%";try{await E.load(g),B()}catch(A){let k=navigator.gpu?A.message:"WebGPU not supported. Use Chrome 113+ or Edge 113+.";throw L(k),P&&P(A),A}}async function et(g){C(),R.style.width="0%";try{await E.loadParts(g),B()}catch(A){let k=navigator.gpu?A.message:"WebGPU not supported. Use Chrome 113+ or Edge 113+.";throw L(k),P&&P(A),A}}async function st(g){try{let A=await E.loadAnimationUrl(g);return Nt(A.callouts),console.log(`[HoloSplat] animation loaded: ${A.frameCount} frames @ ${A.fps}fps, ${A.callouts.length} callout(s):`,A.callouts.map(k=>k.id),"| markers:",A.markers),A}catch(A){console.error("[HoloSplat] animation failed to load:",A),P&&P(A)}}let _t={load:mt,loadParts:et,loadAnim:st,destroy(){E.destroy(),e.innerHTML="",e.classList.remove("hs-player")},setBackground(g){E.setBackground(g)},setSplatScale(g){E.setSplatScale(g)},setAutoRotate(g){E.setAutoRotate(g)},setFlipY(g){E.setFlipY(g)},setAnimationPaused(g){E.setAnimationPaused(g)},setCameraFree(g){E.setCameraFree(g)},resetCamera(){E.resetCamera()},callout(g){return H[g]?.card??null},get camera(){return E.camera},get animation(){return E._animation},get animationPaused(){return E._animPaused}};if(window.__hsPlayers||(window.__hsPlayers=[]),window.__hsPlayers.push({root:e,api:_t,viewer:E}),E.init().then(async()=>{if(E.start(),m&&f){let g=await st(f);if(g?.objects.length){let A=m.replace(/\/?$/,"/"),k=Object.fromEntries(g.objects.map(q=>[q.id,`${A}${lt(q.id)}${l}`]));await et(k)}}else{let g=[];c?g.push(et(c)):U&&g.push(mt(U)),f&&g.push(st(f)),await Promise.all(g)}B(),F?.()}).catch(g=>{navigator.gpu||L("WebGPU not supported. Use Chrome 113+ or Edge 113+.")}),typeof location<"u"&&new URLSearchParams(location.search).has("hs")&&!document.getElementById("__hs-script")){let g=document.createElement("script");g.id="__hs-script",g.src="/holosplat/editor.js",document.head.appendChild(g)}return _t}function Lt(){document.querySelectorAll("[data-holosplat]").forEach(i=>{if(i._hsPlayer)return;let t=i.getAttribute("data-holosplat")||void 0,e=i.getAttribute("data-holosplat-anim")||void 0,s=i.getAttribute("data-holosplat-parts")||void 0;i._hsPlayer=qt(i,{src:t,animation:e,partsDir:s})})}typeof document<"u"&&(document.readyState==="loading"?document.addEventListener("DOMContentLoaded",Lt):Lt());var le=`
.hs-scene {
  position: relative;
  width: 100%;
}
/* .hs-stage also carries .hs-player (position:relative) from player.js.
   !important ensures sticky wins regardless of injection order.
   100vw fills the full viewport regardless of parent padding/margin. */
.hs-stage {
  position: sticky !important;
  top: 0;
  left: 0;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 1;
  overflow: hidden;
}
.hs-stage canvas {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  display: block;
}
.hs-track {
  /* Overlap the sticky stage so acts start scrolling at the same time the
     canvas appears \u2014 not after it. */
  margin-top: -100vh;
  position: relative;
  /* Clicks pass through to the canvas; restore per-element as needed. */
  pointer-events: none;
  z-index: 2;
}
.hs-act,
.hs-hold {
  position: relative;
}
.hs-caption {
  pointer-events: auto;
}
.hs-caption--hidden {
  opacity: 0;
  pointer-events: none;
}
`,Ot=!1;function he(){if(Ot||typeof document>"u")return;Ot=!0;let i=document.createElement("style");i.textContent=le,document.head.appendChild(i)}function ut(i,t,e=0){if(i==null||i==="")return e;let s=String(i).trim();if(Object.prototype.hasOwnProperty.call(t,s))return t[s];let n=parseFloat(s);return isNaN(n)?(console.warn(`[HoloSplat] scrollScene: unknown marker "${s}" \u2014 using ${e}`),e):n}function fe(i,t,e){if(i.classList.contains("hs-hold"))return{el:i,type:"hold",frame:ut(i.dataset.frame,t,0),captions:It(i)};let n=i.dataset.from??"",r=n==="pingpong-start",o=n==="freecamera-start",a=ut(n,t,0),c=ut(i.dataset.to,t,e),f=0;if(i.dataset.loop!==void 0){let m=i.dataset.loop.trim();f=m===""||m==="true"?1:Math.max(1,parseFloat(m)||1)}return{el:i,type:r?"pingpong":o?"freecamera":"act",from:a,to:c,loop:f,captions:It(i)}}function It(i){return[...i.querySelectorAll(".hs-caption")].map(t=>({el:t,at:parseFloat(t.dataset.at??"0")}))}function Dt(i,t){if(i.type==="hold")return i.frame;if(i.type==="pingpong"||i.type==="freecamera")return t>=1?i.to:i.from;let{from:e,to:s,loop:n}=i,r=s-e;if(r===0)return e;let o=t;return n>0&&(o=t*n%1),e+o*r}function ue(i){let t=i.getBoundingClientRect().top,e=i.offsetHeight||1;return Math.max(0,Math.min(1,-t/e))}function de(i,t,e){return i+(t-i)*e}function pe(i,t,e){let s=Math.max(0,Math.min(1,(e-i)/(t-i)));return s*s*(3-2*s)}function Gt(i,t,e={}){he();let s=i.querySelector(".hs-track");if(!s)return console.warn("[HoloSplat] scrollScene: .hs-track not found inside scene"),{rebuild(){},destroy(){}};let n=[],r=!1,o=1,a=0,c=0,f=0,m=null,l=null,u=e.pingpongTransition??.25,h=!1,d=0,p=!1;function _(y,x,w=!1){r||(h=!1,r=!0,c=y,f=x,o=w?-1:1,a=w?x:y,m=null,l=requestAnimationFrame(M))}function v(){r=!1,l!==null&&(cancelAnimationFrame(l),l=null)}function M(y){if(!r)return;if(l=requestAnimationFrame(M),m===null){m=y;return}let x=(y-m)/1e3;m=y;let w=t.animation?.fps??24;a+=o*w*x,a>=f&&(a=f,o=-1),a<=c&&(a=c,o=1),t.animation.seekFrame(a)}function F(){let y=t.animation,x=y?.markers??{},w=y?y.frameCount-1:0;n=[...s.children].map(T=>fe(T,x,w)),y&&t.setAnimationPaused(!0)}function b(){if(!t.animation||!n.length)return;let y=Dt(n[0],0),x=n[0],w=0;for(let S of n){let C=ue(S.el);if(C<=0||(y=Dt(S,C),x=S,w=C,C<1))break}let T=x.type==="pingpong"&&w>0&&w<1,G=x.type==="freecamera"&&w>0&&w<1;if(T?(h=!1,r||_(x.from,x.to,w>.5)):(r&&(d=a,h=!0),v()),G?p||(p=!0,t.setCameraFree(!0),t.camera.enabled=!0,t.camera.panEnabled=!1):p&&(p=!1,t.setCameraFree(!1),t.camera.enabled=!1,t.camera.panEnabled=!0),!T&&!G)if(h){let S=pe(0,u,w);S>=1&&(h=!1),t.animation.seekFrame(de(d,y,S))}else t.animation.seekFrame(y);let R=x?.el?.id??"\u2014";if(b._activeId!==R){b._activeId=R;let S=x,C=S.type==="hold"?`hold @ frame ${S.frame}`:S.type==="pingpong"?`pingpong [${S.from} \u2194 ${S.to}]`:`act [${S.from} \u2192 ${S.to}]`;console.log(`[HoloSplat] active: ${R} (${C}) | current frame: ${y.toFixed(1)}`)}for(let S of n){if(!S.captions.length)continue;let C=S===x;for(let B of S.captions){let L=C&&w>=B.at;B.el.classList.toggle("hs-caption--hidden",!L)}}}let P=!1;function U(){P||(P=!0,requestAnimationFrame(()=>{P=!1,b()}))}return window.addEventListener("scroll",U,{passive:!0}),function y(){if(!t.animation){requestAnimationFrame(y);return}t.camera.enabled=!1,e.onReady&&e.onReady(t.animation),F(),b()}(),{rebuild(){v(),h=!1,p&&(p=!1,t.setCameraFree(!1),t.camera.panEnabled=!0),F(),b()},destroy(){v(),h=!1,p&&(p=!1,t.setCameraFree(!1),t.camera.panEnabled=!0),window.removeEventListener("scroll",U),t.camera.enabled=!0,t.setAnimationPaused(!1)}}}function jt(){document.querySelectorAll(".hs-scene").forEach(i=>{if(i._hsScroll)return;let t=i.querySelector(".hs-stage");t&&t._hsPlayer&&(i._hsScroll=Gt(i,t._hsPlayer))})}typeof document<"u"&&(document.readyState==="loading"?document.addEventListener("DOMContentLoaded",jt):jt());async function me(i,t,e={}){let s=Yt(i,t,e);return ge(s)}function Yt(i,t,e={}){let{fractionalBits:s}=e;if(s==null){let p=0;for(let v=0;v<t;v++){let M=v*16;Math.abs(i[M])>p&&(p=Math.abs(i[M])),Math.abs(i[M+1])>p&&(p=Math.abs(i[M+1])),Math.abs(i[M+2])>p&&(p=Math.abs(i[M+2]))}let _=(1<<23)-1;s=p>0?Math.min(20,Math.max(0,Math.floor(Math.log2(_/p)))):12}let n=16,r=n+t*20,o=new ArrayBuffer(r),a=new DataView(o),c=new Uint8Array(o);a.setUint32(0,1347635022,!0),a.setUint32(4,3,!0),a.setUint32(8,t,!0),a.setUint8(12,0),a.setUint8(13,s),a.setUint8(14,0),a.setUint8(15,0);let f=n,m=f+t*9,l=m+t*1,u=l+t*3,h=u+t*3,d=1<<s;for(let p=0;p<t;p++){let _=p*16;dt(a,f+p*9+0,i[_+0]*d),dt(a,f+p*9+3,i[_+1]*d),dt(a,f+p*9+6,i[_+2]*d),c[m+p]=X(i[_+7]*255),c[l+p*3+0]=X(i[_+4]*255),c[l+p*3+1]=X(i[_+5]*255),c[l+p*3+2]=X(i[_+6]*255),c[u+p*3+0]=pt(i[_+8]),c[u+p*3+1]=pt(i[_+9]),c[u+p*3+2]=pt(i[_+10]),a.setUint32(h+p*4,_e(i[_+12],i[_+13],i[_+14],i[_+15]),!0)}return c}function dt(i,t,e){let s=Math.max(-8388608,Math.min(8388607,Math.round(e)));i.setUint8(t,s&255),i.setUint8(t+1,s>>8&255),i.setUint8(t+2,s>>16&255)}function X(i){return Math.max(0,Math.min(255,Math.round(i)))}function pt(i){return X(Math.log(Math.max(1e-9,i))*16+128)}function _e(i,t,e,s){let n=Math.hypot(i,t,e,s)||1,r=[i/n,t/n,e/n,s/n],o=0;for(let d=1;d<4;d++)Math.abs(r[d])>Math.abs(r[o])&&(o=d);let a=r[o]<0?-1:1,c=[0,1,2,3].filter(d=>d!==o),f=512*Math.SQRT2,m=d=>Math.max(-512,Math.min(511,Math.round(r[d]*a*f))),l=m(c[0]),u=m(c[1]),h=m(c[2]);return(o&3|(l&1023)<<2|(u&1023)<<12|(h&1023)<<22)>>>0}async function ge(i){if(typeof CompressionStream>"u")throw new Error("CompressionStream API is not available in this environment");let t=new CompressionStream("gzip"),e=t.writable.getWriter();return e.write(i),e.close(),new Response(t.readable).arrayBuffer()}async function Qe(i={}){let{onLoad:t,onError:e,src:s,parts:n,...r}=i,o=new $({...r}),a={destroy(){},setBackground(){},setSplatScale(){},setAutoRotate(){},resetCamera(){},camera:null};try{await o.init(),n?await o.loadParts(n):s&&await o.load(s)}catch(c){if(o.destroy(),e)return e(c),a;throw c}return o.start(),t?.(),{destroy(){o.destroy()},setBackground(c){o.setBackground(c)},setSplatScale(c){o.setSplatScale(c)},setAutoRotate(c){o.setAutoRotate(c)},setFlipY(c){o.setFlipY(c)},resetCamera(){o.resetCamera()},get camera(){return o.camera}}}export{K as Animation,$ as Viewer,me as compressToSpz,Qe as create,Yt as encodeSpz,ht as loadAnimation,Ut as parsePly,Pt as parseSplat,qt as player,Gt as scrollScene,lt as splatNameFromId};
//# sourceMappingURL=holosplat.esm.js.map
