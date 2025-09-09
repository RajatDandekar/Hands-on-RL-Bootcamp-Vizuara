import React, { useEffect, useMemo, useState } from "react";

/**
 * Reward Model Training — v2.2 (Masks + Block Size + Clear Vectors)
 * -----------------------------------------------------------------
 * Adds:
 *  • Full masking + block_size pipeline (pad/truncate with EOT, build masks)
 *  • Mask inclusion visuals (colored grids + active% bars)
 *  • Clearer vector view (bigger canvas, arrowheads, legend)
 *  • Animated training (SGD on L = -logσ(Δ)) and loss/accuracy charts
 *
 * Notes:
 *  • False = active token, True = masked (ignored)
 *  • We compute reward from the last UNMASKED position (matches your code)
 */

// ---------------- Tokenizer & Embeddings (toy) ----------------
function toyTokenize(str) {
  if (!str) return [];
  return str
    .replace(/\n/g, " ")
    .split(/(\s+|[,.!?;:()\[\]\-\"'])/)
    .filter(Boolean)
    .filter((s) => /\S/.test(s));
}
function hash32(s) { let h = 2166136261 >>> 0; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
function toyEmbed(id, dim = 8) { const rng = (seed) => { let x = seed >>> 0; return () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) / 0xffffffff; }; }; const r = rng(id); return new Array(dim).fill(0).map(() => r() * 2 - 1); }

// ---------------- Block-size + mask utilities -----------------
const EOT_ID = 50256;
function buildIdsAndMask(prompt, response, blockSize) {
  // ids = token ids, mask = boolean (False=active, True=masked)
  const pIds = prompt.map((t) => hash32(t) % 50000);
  const rIds = response.map((t) => hash32(t) % 50000);
  const concat = [...pIds, ...rIds];

  if (concat.length >= blockSize) {
    // Truncate from front: keep last blockSize (mask all False)
    const kept = concat.slice(-blockSize);
    const mask = new Array(blockSize).fill(false);
    return { ids: kept, mask, truncated: true, padded: false, activeLen: blockSize, promptLen: pIds.length };
  } else {
    // Pad with EOT; mask[len(concat)+1:] = True (following your pad_toks)
    const ids = Array(blockSize).fill(EOT_ID);
    for (let i = 0; i < concat.length; i++) ids[i] = concat[i];
    const mask = Array(blockSize).fill(false);
    for (let j = concat.length + 1; j < blockSize; j++) mask[j] = true;
    return { ids, mask, truncated: false, padded: true, activeLen: concat.length, promptLen: pIds.length };
  }
}

// --------------- Reward computation pieces -------------------
function toyTransformerFromIds(ids, mask, dim = 8) {
  const T = ids.length;
  const embs = ids.map((id) => toyEmbed(id, dim));
  // last unmasked index
  let lastIdx = T - 1; for (let i = T - 1; i >= 0; i--) { if (!mask[i]) { lastIdx = i; break; } }
  // mean-pool up to lastIdx
  const pooled = new Array(dim).fill(0);
  for (let t = 0; t <= lastIdx; t++) { for (let d = 0; d < dim; d++) pooled[d] += embs[t][d]; }
  const cnt = Math.max(1, lastIdx + 1); for (let d = 0; d < dim; d++) pooled[d] /= cnt;
  return { lastIdx, rep: pooled };
}
function linear(vec, w, b) { let s = b; for (let i = 0; i < vec.length; i++) s += w[i] * vec[i]; return s; }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function logsigmoid(x) { return -Math.log(1 + Math.exp(-x)); }

// ------------------------- UI atoms --------------------------
const Label = ({ children }) => (<div className="text-xs uppercase tracking-wide text-gray-600 mb-1">{children}</div>);
const Pill = ({ children, cls = "bg-slate-100 text-slate-900" }) => (<span className={`px-2 py-0.5 rounded text-xs font-mono ${cls}`}>{children}</span>);

// ----------------------- Main component ----------------------
export default function RewardModelTraining() {
  // Inputs
  const [promptTxt, setPromptTxt] = useState("Summarize: Pune is a vibrant Indian city known for education and culture.");
  const [chosenTxt, setChosenTxt] = useState("Pune is an Indian city celebrated for education and culture.");
  const [rejectTxt, setRejectTxt] = useState("Pune might be somewhere; the details are unclear and repetitive.");
  const [blockSize, setBlockSize] = useState(48);

  // Tokenize strings → tokens
  const pToks = useMemo(() => toyTokenize(promptTxt), [promptTxt]);
  const cToks = useMemo(() => toyTokenize(chosenTxt), [chosenTxt]);
  const rToks = useMemo(() => toyTokenize(rejectTxt), [rejectTxt]);

  // Build ids + masks for POS (prompt+chosen) and NEG (prompt+rejected)
  const posPack = useMemo(() => buildIdsAndMask(pToks, cToks, blockSize), [pToks, cToks, blockSize]);
  const negPack = useMemo(() => buildIdsAndMask(pToks, rToks, blockSize), [pToks, rToks, blockSize]);

  // Transformer representations using masks
  const dim = 8;
  const posOut = useMemo(() => toyTransformerFromIds(posPack.ids, posPack.mask, dim), [posPack, dim]);
  const negOut = useMemo(() => toyTransformerFromIds(negPack.ids, negPack.mask, dim), [negPack, dim]);

  // Feature diff and head
  const dVec = useMemo(() => posOut.rep.map((v, i) => v - negOut.rep[i]), [posOut.rep, negOut.rep]);
  const [w, setW] = useState(() => new Array(dim).fill(0));
  const [b] = useState(0);
  const chReward = useMemo(() => linear(posOut.rep, w, b), [posOut.rep, w, b]);
  const rjReward = useMemo(() => linear(negOut.rep, w, b), [negOut.rep, w, b]);
  const margin = useMemo(() => chReward - rjReward, [chReward, rjReward]);
  const loss = useMemo(() => -logsigmoid(margin), [margin]);
  const acc = useMemo(() => (margin > 0 ? 1 : 0), [margin]);

  // Training state
  const [lr, setLr] = useState(0.5);
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [hist, setHist] = useState([]);
  const gradW = useMemo(() => { const g = sigmoid(margin) - 1; return dVec.map((x) => g * x); }, [margin, dVec]);
  const sgdStep = () => { setW((prev) => prev.map((wi, i) => wi - lr * gradW[i])); setStep((s) => s + 1); setHist((h) => [...h, { step: step + 1, loss, acc, margin }].slice(-200)); };
  useEffect(() => { if (!isPlaying) return; const t = setTimeout(() => sgdStep(), 600); return () => clearTimeout(t); }, [isPlaying, loss, acc, margin, gradW, lr]);
  const reset = () => { setIsPlaying(false); setStep(0); setW(new Array(dim).fill(0)); setHist([]); };

  // ------------------- Visual helpers -------------------
  function MaskGrid({ mask, title }) {
    return (
      <div className="border rounded p-3 bg-white">
        <div className="font-semibold mb-2">{title}</div>
        <div className="grid grid-cols-16 gap-1">
          {mask.map((m, i) => (
            <div key={i} className={`h-4 w-4 rounded ${m ? 'bg-gray-300' : 'bg-emerald-400'}`} title={`${i}: ${m ? 'masked' : 'active'}`} />
          ))}
        </div>
        <div className="text-xs text-gray-500 mt-2"><span className="inline-block h-3 w-3 bg-emerald-400 rounded mr-1" /> active &nbsp; <span className="inline-block h-3 w-3 bg-gray-300 rounded mr-1" /> masked</div>
      </div>
    );
  }
  function InclusionBars() {
    const posActive = posPack.mask.filter((m) => !m).length;
    const negActive = negPack.mask.filter((m) => !m).length;
    const pct = (n) => Math.round((n / blockSize) * 100);
    return (
      <div className="border rounded p-3 bg-white">
        <div className="font-semibold mb-2">Inclusion (Active tokens %)</div>
        <div className="space-y-2">
          <div className="text-xs">POS: {posActive}/{blockSize} ({pct(posActive)}%)</div>
          <div className="h-3 bg-gray-200 rounded"><div className="h-3 bg-emerald-500 rounded" style={{ width: `${(posActive / blockSize) * 100}%` }} /></div>
          <div className="text-xs mt-2">NEG: {negActive}/{blockSize} ({pct(negActive)}%)</div>
          <div className="h-3 bg-gray-200 rounded"><div className="h-3 bg-rose-500 rounded" style={{ width: `${(negActive / blockSize) * 100}%` }} /></div>
        </div>
      </div>
    );
  }

  function LineChart({ data, yKey, title }) {
    const W = 420, H = 160, pad = 28;
    const xs = data.map((d) => d.step); const ys = data.map((d) => d[yKey]);
    const xMin = xs.length ? Math.min(...xs) : 0, xMax = xs.length ? Math.max(...xs) : 1;
    const yMin = ys.length ? Math.min(...ys) : 0, yMax = ys.length ? Math.max(...ys) : 1;
    const X = (x) => pad + (xs.length ? ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (W - 2 * pad) : 0);
    const Y = (y) => H - pad - (ys.length ? ((y - yMin) / Math.max(1e-9, yMax - yMin)) * (H - 2 * pad) : 0);
    const path = data.map((d, i) => `${i === 0 ? 'M' : 'L'}${X(d.step)},${Y(d[yKey])}`).join(' ');
    return (
      <svg width={W} height={H} className="border rounded bg-white">
        <text x={pad} y={18} className="text-[12px]">{title}</text>
        <path d={path} fill="none" stroke="currentColor" />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
      </svg>
    );
  }
  function LossCurveWithMarker() {
    const W = 380, H = 160, pad = 20;
    const xs = Array.from({ length: 101 }, (_, i) => -6 + (12 * i) / 100);
    const ys = xs.map((x) => -logsigmoid(x));
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const X = (x) => pad + ((x + 6) / 12) * (W - 2 * pad);
    const Y = (y) => pad + ((y - yMin) / (yMax - yMin + 1e-9)) * (H - 2 * pad);
    const path = xs.map((x, i) => `${i ? 'L' : 'M'}${X(x)},${Y(ys[i])}`).join(' ');
    return (
      <svg width={W} height={H} className="border rounded bg-white">
        <text x={pad} y={14} className="text-[12px]">Loss = −logσ(Δ)</text>
        <path d={path} fill="none" stroke="currentColor" />
        <circle cx={X(margin)} cy={Y(-logsigmoid(margin))} r={3} />
        <text x={X(margin) + 6} y={Y(-logsigmoid(margin)) - 6} className="text-[10px]">Δ={margin.toFixed(2)}</text>
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#ccc" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#ccc" />
      </svg>
    );
  }

  // Schematic of forward + loss + grad
  function Schematic() {
    return (
      <div className="grid md:grid-cols-3 gap-4">
        <div className="border rounded p-3 bg-white">
          <div className="font-semibold mb-1">1) Build inputs (block_size)</div>
          <div className="text-xs text-gray-600">Concat prompt + response → pad with EOT or truncate to length = block_size. Create mask where indices &gt;= len(concat)+1 are masked.</div>
          <div className="mt-2 text-xs">POS active: <Pill>{posPack.activeLen}</Pill> / NEG active: <Pill cls="bg-rose-100 text-rose-900">{negPack.activeLen}</Pill></div>
        </div>
        <div className="border rounded p-3 bg-white">
          <div className="font-semibold mb-1">2) Rewards</div>
          <div className="text-xs text-gray-600">Transformer → pick last unmasked hidden state → linear head r = w·h + b</div>
          <div className="mt-2 text-sm">r(rej) = <Pill cls="bg-rose-100 text-rose-900">{rjReward.toFixed(3)}</Pill></div>
          <div className="text-sm">r(ch) = <Pill cls="bg-emerald-100 text-emerald-900">{chReward.toFixed(3)}</Pill></div>
        </div>
        <div className="border rounded p-3 bg-white">
          <div className="font-semibold mb-1">3) Loss & Gradient</div>
          <div className="font-mono text-xs">Δ = w·(h_pos − h_neg)</div>
          <div className="font-mono text-xs">L = −logσ(Δ),  ∂L/∂w = (σ(Δ) − 1) · (h_pos − h_neg)</div>
          <div className="mt-2 text-sm">Δ = <Pill cls="bg-amber-100 text-amber-900">{margin.toFixed(3)}</Pill> &nbsp; L = <Pill>{loss.toFixed(4)}</Pill> &nbsp; Acc = <Pill cls={acc? 'bg-emerald-100 text-emerald-900':'bg-gray-200'}>{acc}</Pill></div>
        </div>
      </div>
    );
  }

  // Clear vector view (2D slice): d and -grad
  function VectorView() {
    const W = 460, H = 220, cx = W/2, cy = H/2, scale = 60;
    const d = [dVec[0] || 0, dVec[1] || 0];
    const g = [-(gradW[0] || 0), -(gradW[1] || 0)]; // descent
    const end = (v) => ({ x: cx + scale * v[0], y: cy - scale * v[1] });
    const dE = end(d), gE = end(g);
    return (
      <svg width={W} height={H} className="border rounded bg-white">
        <defs>
          <marker id="arrow-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L6,3 L0,6 z" fill="#10b981" />
          </marker>
          <marker id="arrow-red" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L6,3 L0,6 z" fill="#ef4444" />
          </marker>
        </defs>
        <text x={10} y={16} className="text-[12px]">Vector view</text>
        {/* axes */}
        <line x1={0} y1={cy} x2={W} y2={cy} stroke="#ddd" />
        <line x1={cx} y1={0} x2={cx} y2={H} stroke="#ddd" />
        {/* d = h_pos - h_neg */}
        <line x1={cx} y1={cy} x2={dE.x} y2={dE.y} stroke="#10b981" strokeWidth={2} markerEnd="url(#arrow-green)" />
        {/* -grad direction */}
        <line x1={cx} y1={cy} x2={gE.x} y2={gE.y} stroke="#ef4444" strokeWidth={2} markerEnd="url(#arrow-red)" />
        {/* legend */}
        <rect x={W-160} y={10} width={150} height={44} rx={6} fill="#f8fafc" stroke="#e2e8f0" />
        <circle cx={W-146} cy={24} r={4} fill="#10b981" /><text x={W-136} y={28} className="text-[11px]">d = h_pos − h_neg</text>
        <circle cx={W-146} cy={44} r={4} fill="#ef4444" /><text x={W-136} y={48} className="text-[11px]">descent (−∇w L)</text>
      </svg>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-4 sm:p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow p-4 sm:p-6">
        <h1 className="text-2xl sm:text-3xl font-bold mb-2">Reward Model Training — Masks, Block Size & Animated Learning</h1>
        <p className="text-gray-600 mb-6">We build (prompt + response) with block_size, pad/truncate with EOT and a mask, pick the last unmasked hidden state for rewards, and train with a pairwise objective. Watch loss fall and accuracy rise.</p>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <button onClick={() => setIsPlaying((v) => !v)} className={`px-3 py-2 rounded text-white ${isPlaying ? 'bg-red-600' : 'bg-green-600'}`}>{isPlaying ? 'Pause' : 'Play'}</button>
          <button onClick={sgdStep} disabled={isPlaying} className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50">Step</button>
          <button onClick={reset} className="px-3 py-2 rounded bg-gray-600 text-white">Reset</button>
          <div className="flex items-center gap-2 ml-4"><Label>LR</Label><input type="range" min={0.05} max={1.0} step={0.05} value={lr} onChange={(e)=>setLr(parseFloat(e.target.value))} /><Pill>{lr.toFixed(2)}</Pill></div>
          <div className="flex items-center gap-2 ml-2"><Label>Step</Label><Pill>{step}</Pill></div>
          <div className="flex items-center gap-2 ml-4"><Label>block_size</Label><input type="range" min={16} max={128} step={1} value={blockSize} onChange={(e)=>setBlockSize(parseInt(e.target.value))} /><Pill>{blockSize}</Pill></div>
        </div>

        {/* Text inputs */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div><Label>Prompt</Label><textarea value={promptTxt} onChange={(e)=>setPromptTxt(e.target.value)} className="w-full border rounded p-2 h-24" /></div>
          <div><Label>Chosen (positive)</Label><textarea value={chosenTxt} onChange={(e)=>setChosenTxt(e.target.value)} className="w-full border rounded p-2 h-24" /></div>
          <div><Label>Rejected (negative)</Label><textarea value={rejectTxt} onChange={(e)=>setRejectTxt(e.target.value)} className="w-full border rounded p-2 h-24" /></div>
        </div>

        {/* Masks + inclusion graphs */}
        <div className="grid lg:grid-cols-3 gap-6 mb-6">
          <MaskGrid mask={posPack.mask} title="POS mask (prompt + chosen)" />
          <MaskGrid mask={negPack.mask} title="NEG mask (prompt + rejected)" />
          <InclusionBars />
        </div>

        {/* Schematic + curves + vectors */}
        <div className="grid lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-2 space-y-4">
            <Schematic />
            <div className="grid sm:grid-cols-2 gap-4">
              <LossCurveWithMarker />
              <VectorView />
            </div>
          </div>
          <div className="space-y-3">
            <div className="border rounded p-3 bg-white">
              <div className="font-semibold mb-1">Current state</div>
              <div className="text-sm">r(rej): <Pill cls="bg-rose-100 text-rose-900">{rjReward.toFixed(3)}</Pill></div>
              <div className="text-sm">r(ch): <Pill cls="bg-emerald-100 text-emerald-900">{chReward.toFixed(3)}</Pill></div>
              <div className="text-sm">Δ: <Pill cls="bg-amber-100 text-amber-900">{margin.toFixed(3)}</Pill> &nbsp; L: <Pill>{loss.toFixed(4)}</Pill> &nbsp; Acc: <Pill cls={acc? 'bg-emerald-100 text-emerald-900':'bg-gray-200'}>{acc}</Pill></div>
              <div className="text-xs text-gray-500 mt-1">Last unmasked index — POS: <Pill>{posOut.lastIdx}</Pill> &nbsp; NEG: <Pill cls="bg-rose-100 text-rose-900">{negOut.lastIdx}</Pill></div>
            </div>
            <div className="border rounded p-3 bg-white">
              <div className="font-semibold mb-1">Head weights w</div>
              <div className="flex flex-wrap gap-2">
                {w.map((wi, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <Pill> w[{i}] </Pill>
                    <input type="number" step="0.05" value={wi} onChange={(e)=>{ const vv = [...w]; vv[i] = parseFloat(e.target.value); setW(vv); }} className="w-20 border rounded p-1 text-xs font-mono" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Training curves */}
        <div className="grid md:grid-cols-2 gap-6">
          <LineChart data={hist} yKey="loss" title="Loss vs Steps" />
          <LineChart data={hist} yKey="acc" title="Accuracy vs Steps" />
        </div>

        {/* Notes */}
        <div className="mt-6 text-sm text-gray-700 space-y-2">
          <div><span className="font-semibold">Masking:</span> False = active, True = masked. We compute the reward from the last False index (closest to the end).</div>
          <div><span className="font-semibold">Padding:</span> If length &lt; block_size, pad with EOT ({EOT_ID}) and set mask[i] = True for i &gt;= len(concat)+1. If length &ge; block_size, keep the last block_size tokens (mask all False).</div>
          <div><span className="font-semibold">Training:</span> Δ grows as w aligns with d = h_pos − h_neg; bigger Δ → larger σ(Δ) → lower −logσ(Δ) and Acc tends to 1.</div>
        </div>
      </div>
    </div>
  );
}
