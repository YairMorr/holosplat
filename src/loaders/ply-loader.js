/**
 * Loader for 3D Gaussian Splatting .ply files (standard 3DGS training output).
 *
 * Parses the ASCII header to discover property names and their byte offsets,
 * then reads the binary body. Handles all the 3DGS-specific transforms:
 *   - f_dc_0/1/2 → RGB via:  0.5 + SH_C0 * f_dc  (SH_C0 = 0.28209479177387814)
 *   - opacity    → sigmoid: 1 / (1 + exp(−x))
 *   - scale_i    → linear:  exp(scale_i)
 *   - rot_0..3   → quaternion, rot_0 = w (swizzle to xyzw), normalize
 *
 * Higher-order SH coefficients (f_rest_*) are ignored.
 *
 * Output: same canonical Float32Array layout as splat-loader (16 floats/Gaussian).
 */

const SH_C0 = 0.28209479177387814;

export async function loadPly(url, onProgress) {
  const buffer = await fetchWithProgress(url, onProgress);
  return parsePly(buffer);
}

export function parsePly(buffer) {
  const { numVertices, properties, dataOffset } = parseHeader(buffer);
  if (numVertices === 0) throw new Error('PLY file contains no vertices');

  // Build a property lookup: name → { byteOffset, type }
  const propMap = {};
  let stride = 0;
  for (const p of properties) {
    propMap[p.name] = { offset: stride, type: p.type };
    stride += sizeOf(p.type);
  }

  // Check required fields exist
  const required = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2',
                    'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity'];
  for (const r of required) {
    if (!propMap[r]) throw new Error(`PLY missing required property: ${r}`);
  }

  const hasColor = propMap['f_dc_0'] && propMap['f_dc_1'] && propMap['f_dc_2'];

  const src  = new DataView(buffer, dataOffset);
  const data = new Float32Array(numVertices * 16);

  for (let i = 0; i < numVertices; i++) {
    const base = i * stride;
    const d    = i * 16;

    // Position
    data[d + 0] = read(src, base, propMap['x']);
    data[d + 1] = read(src, base, propMap['y']);
    data[d + 2] = read(src, base, propMap['z']);
    // d[3] = 0

    // Color (DC spherical harmonics → linear RGB) or white fallback
    if (hasColor) {
      data[d + 4] = clamp01(0.5 + SH_C0 * read(src, base, propMap['f_dc_0']));
      data[d + 5] = clamp01(0.5 + SH_C0 * read(src, base, propMap['f_dc_1']));
      data[d + 6] = clamp01(0.5 + SH_C0 * read(src, base, propMap['f_dc_2']));
    } else {
      data[d + 4] = 1; data[d + 5] = 1; data[d + 6] = 1;
    }
    // Opacity: logit-space → sigmoid
    data[d + 7] = sigmoid(read(src, base, propMap['opacity']));

    // Scale: log-space → linear
    data[d +  8] = Math.exp(read(src, base, propMap['scale_0']));
    data[d +  9] = Math.exp(read(src, base, propMap['scale_1']));
    data[d + 10] = Math.exp(read(src, base, propMap['scale_2']));
    // d[11] = 0

    // Quaternion: PLY stores (w, x, y, z) as rot_0..3; we need (x, y, z, w)
    const rw = read(src, base, propMap['rot_0']);
    const rx = read(src, base, propMap['rot_1']);
    const ry = read(src, base, propMap['rot_2']);
    const rz = read(src, base, propMap['rot_3']);
    const rl = Math.hypot(rx, ry, rz, rw) || 1;
    data[d + 12] = rx / rl;
    data[d + 13] = ry / rl;
    data[d + 14] = rz / rl;
    data[d + 15] = rw / rl;
  }

  return { data, count: numVertices };
}

// ── Header parsing ─────────────────────────────────────────────────────────

function parseHeader(buffer) {
  // Read bytes until we find "end_header\n"
  const bytes   = new Uint8Array(buffer);
  const END_TAG = 'end_header';
  let headerEnd = -1;
  let text      = '';

  for (let i = 0; i < bytes.length; i++) {
    text += String.fromCharCode(bytes[i]);
    if (text.endsWith(END_TAG)) {
      // Skip the trailing newline(s)
      headerEnd = i + 1;
      if (bytes[headerEnd] === 13) headerEnd++; // \r
      if (bytes[headerEnd] === 10) headerEnd++; // \n
      break;
    }
  }

  if (headerEnd < 0) throw new Error('Invalid PLY: end_header not found');

  const lines = text.split('\n');
  let numVertices = 0;
  const properties = [];
  let inVertex = false;

  for (const raw of lines) {
    const line  = raw.trim();
    const parts = line.split(/\s+/);
    if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      if (inVertex) numVertices = parseInt(parts[2], 10);
    } else if (parts[0] === 'property' && inVertex) {
      properties.push({ type: parts[1], name: parts[2] });
    }
  }

  return { numVertices, properties, dataOffset: headerEnd };
}

// ── Binary read helpers ────────────────────────────────────────────────────

function sizeOf(type) {
  switch (type) {
    case 'float': case 'float32': case 'int':   case 'uint':   return 4;
    case 'double': case 'int64': case 'uint64': return 8;
    case 'short': case 'ushort': case 'int16': case 'uint16': return 2;
    case 'char':  case 'uchar':  case 'int8':  case 'uint8':  return 1;
    default: return 4; // safe fallback
  }
}

function read(view, base, prop) {
  const off = base + prop.offset;
  switch (prop.type) {
    case 'float': case 'float32': return view.getFloat32(off, true);
    case 'double':                return view.getFloat64(off, true);
    case 'int': case 'int32':     return view.getInt32(off, true);
    case 'uint': case 'uint32':   return view.getUint32(off, true);
    case 'short': case 'int16':   return view.getInt16(off, true);
    case 'ushort': case 'uint16': return view.getUint16(off, true);
    case 'char': case 'int8':     return view.getInt8(off);
    case 'uchar': case 'uint8':   return view.getUint8(off);
    default:                      return view.getFloat32(off, true);
  }
}

// ── Math helpers ───────────────────────────────────────────────────────────

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// fetchWithProgress is defined in fetch-utils.js (shared)
