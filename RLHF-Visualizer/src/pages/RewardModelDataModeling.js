import React, { useMemo, useState } from "react";

/**
 * RLHF Pairwise Dataset Visualization
 * -----------------------------------
 * Explains the snippet:
 *   - Filtering long (prompt+chosen)/(prompt+rejected) pairs by block_size
 *   - Padding with EOT and building a boolean mask (pad_toks)
 *   - __getitem__ that returns pos/neg tensors and masks
 *
 * Uses a toy tokenizer (hash-based ids) so the demo is self-contained.
 * You can paste your own prompt/chosen/rejected and play with block_size.
 */

// ------- Toy Tokenizer (stand-in for BPETokenizer) -------
const EOT_ID = 50256; // commonly used as an end-of-text id in GPT-2 like tokenizers
function toyTokenize(text) {
  // split on whitespace but keep punctuation blocks separate for visibility
  if (!text) return [];
  const parts = text
    .replace(/\n/g, " ")
    .split(/(\s+|[,.!?;:()\[\]\-\"'])/)
    .filter(Boolean)
    .filter((s) => /\S/.test(s));
  return parts.map((t) => ({
    tok: t,
    id: Math.abs(hash32(t)) % 50000, // fake id, under 50k to visualize
  }));
}
function hash32(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// ------- Helpers matching the Python logic -------
function padToks(ids, blockSize) {
  // Returns { toks: number[blockSize], mask: boolean[blockSize], cutFromFront: boolean }
  let toks = [...ids];
  const mask = Array(blockSize).fill(false);
  let cutFromFront = false;

  if (toks.length >= blockSize) {
    // keep the LAST blockSize tokens (Python: toks = toks[-block_size:])
    toks = toks.slice(-blockSize);
    cutFromFront = true;
  } else {
    const pad = Array(blockSize).fill(EOT_ID);
    for (let i = 0; i < toks.length; i++) pad[i] = toks[i];
    // include a final eot token to predict → mask[len(toks) + 1:] = True
    // i.e., everything AFTER the next position is masked out from loss
    for (let j = (toks.length + 1); j < blockSize; j++) mask[j] = true;
    toks = pad;
  }
  return { toks, mask, cutFromFront };
}

function concatIds(promptIds, respIds) {
  return [...promptIds, ...respIds];
}

// ------- Small UI atoms -------
const Label = ({ children }) => (
  <div className="text-xs uppercase tracking-wide text-gray-600 mb-1">{children}</div>
);

const Pill = ({ children, color = "bg-slate-100 text-slate-800" }) => (
  <span className={`px-2 py-0.5 rounded text-xs font-mono ${color}`}>{children}</span>
);

// ------- Main component -------
export default function PairwiseRewardDataViz() {
  const [blockSize, setBlockSize] = useState(32);
  const [prompt, setPrompt] = useState("Summarize the following article in one sentence: Pune is a vibrant city in India, known for education and culture.");
  const [chosen, setChosen] = useState("Pune is a major Indian city celebrated for its education and culture.");
  const [rejected, setRejected] = useState("Pune is somewhere; I don't know much about it, maybe a place with something happening.");

  // Tokenize (toy)
  const promptToks = useMemo(() => toyTokenize(prompt), [prompt]);
  const chosenToks = useMemo(() => toyTokenize(chosen), [chosen]);
  const rejectedToks = useMemo(() => toyTokenize(rejected), [rejected]);

  const promptLen = promptToks.length;
  const chosenLen = chosenToks.length;
  const rejectedLen = rejectedToks.length;

  // Filter rule (Python): keep only if prompt+chosen <= block_size+1 AND prompt+rejected <= block_size+1
  const keepPair = (promptLen + chosenLen <= blockSize + 1) && (promptLen + rejectedLen <= blockSize + 1);

  // __getitem__ path (if kept)
  const posConcat = useMemo(() => concatIds(promptToks.map(t => t.id), chosenToks.map(t => t.id)), [promptToks, chosenToks]);
  const negConcat = useMemo(() => concatIds(promptToks.map(t => t.id), rejectedToks.map(t => t.id)), [promptToks, rejectedToks]);

  const posPadded = useMemo(() => padToks(posConcat, blockSize), [posConcat, blockSize]);
  const negPadded = useMemo(() => padToks(negConcat, blockSize), [negConcat, blockSize]);

  const MaskViz = ({ mask }) => (
    <div className="grid grid-cols-16 gap-1">
      {mask.map((m, i) => (
        <div key={i} className={`h-5 w-5 rounded ${m ? 'bg-gray-300' : 'bg-emerald-400'}`} title={`${i}: ${m ? 'masked (ignored)' : 'kept (active)'}`} />
      ))}
    </div>
  );

  const TokensRow = ({ ids, focusLen }) => (
    <div className="flex flex-wrap gap-1">
      {ids.map((id, i) => (
        <Pill key={i} color={i < focusLen ? 'bg-blue-100 text-blue-900' : 'bg-green-100 text-green-900'}>{id}</Pill>
      ))}
    </div>
  );

  const TextTokens = ({ toks, title }) => (
    <div className="border rounded p-3 bg-white">
      <Label>{title}</Label>
      <div className="flex flex-wrap gap-1">
        {toks.map((t, i) => (
          <Pill key={i}>{t.tok}</Pill>
        ))}
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-4 sm:p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow p-4 sm:p-6">
        <h1 className="text-2xl sm:text-3xl font-bold mb-2">RLHF Pairwise Dataset — Interactive Walkthrough</h1>
        <p className="text-gray-600 mb-6">Explore how long examples are dropped, how sequences are padded with EOT, and how masks are constructed for positive (chosen) and negative (rejected) samples.</p>

        {/* Controls */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div>
            <Label>block_size</Label>
            <div className="flex items-center gap-3">
              <input type="range" min={8} max={128} step={1} value={blockSize} onChange={(e) => setBlockSize(parseInt(e.target.value))} className="w-full" />
              <Pill>{blockSize}</Pill>
            </div>
            <div className="text-xs text-gray-500 mt-1">Filter condition requires: len(prompt)+len(resp) ≤ block_size + 1</div>
          </div>
          <div>
            <Label>split</Label>
            <select className="w-full border rounded p-2 text-sm bg-slate-50">
              <option>train</option>
              <option>validation</option>
              <option>test</option>
            </select>
            <div className="text-xs text-gray-500 mt-1">(Illustrative only)</div>
          </div>
          <div className="border rounded p-3 bg-amber-50">
            <div className="font-semibold text-amber-900">Filter result</div>
            <div className="text-sm">{keepPair ? "✅ Kept" : "❌ Dropped"}</div>
            <div className="text-xs text-amber-900 mt-1">Needs both (prompt+chosen) and (prompt+rejected) ≤ block_size + 1</div>
          </div>
        </div>

        {/* Inputs */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div>
            <Label>Prompt</Label>
            <textarea value={prompt} onChange={(e)=>setPrompt(e.target.value)} className="w-full border rounded p-2 h-28" />
          </div>
          <div>
            <Label>Chosen (positive)</Label>
            <textarea value={chosen} onChange={(e)=>setChosen(e.target.value)} className="w-full border rounded p-2 h-28" />
          </div>
          <div>
            <Label>Rejected (negative)</Label>
            <textarea value={rejected} onChange={(e)=>setRejected(e.target.value)} className="w-full border rounded p-2 h-28" />
          </div>
        </div>

        {/* Tokenization preview */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <TextTokens toks={promptToks} title={`Prompt tokens (${promptLen})`} />
          <TextTokens toks={chosenToks} title={`Chosen tokens (${chosenLen})`} />
          <TextTokens toks={rejectedToks} title={`Rejected tokens (${rejectedLen})`} />
        </div>

        {/* Concat + pad/mask views */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* POSITIVE PATH */}
          <div className="border rounded-lg p-4 bg-white">
            <h3 className="font-semibold mb-2">Positive sample (prompt + chosen)</h3>
            <Label>Concatenated ids</Label>
            <TokensRow ids={posConcat} focusLen={promptLen} />
            <div className="text-xs text-gray-500 mt-1">Blue = prompt segment, Green = chosen segment</div>
            <div className="mt-3 grid gap-2">
              <div>
                <Label>Padded toks (length = block_size)</Label>
                <TokensRow ids={posPadded.toks} focusLen={Math.min(posConcat.length, blockSize)} />
                {posPadded.cutFromFront && (
                  <div className="text-xs text-amber-700 mt-1">⚠️ Overlength — kept last {blockSize} tokens (front truncated).</div>
                )}
              </div>
              <div>
                <Label>Mask (False = active, True = ignore)</Label>
                <MaskViz mask={posPadded.mask} />
                <div className="text-xs text-gray-500 mt-1">Following the code: indices &gt;= len(concat) + 1 are masked (when padding case).</div>
              </div>
            </div>
          </div>

          {/* NEGATIVE PATH */}
          <div className="border rounded-lg p-4 bg-white">
            <h3 className="font-semibold mb-2">Negative sample (prompt + rejected)</h3>
            <Label>Concatenated ids</Label>
            <TokensRow ids={negConcat} focusLen={promptLen} />
            <div className="text-xs text-gray-500 mt-1">Blue = prompt segment, Green = rejected segment</div>
            <div className="mt-3 grid gap-2">
              <div>
                <Label>Padded toks (length = block_size)</Label>
                <TokensRow ids={negPadded.toks} focusLen={Math.min(negConcat.length, blockSize)} />
                {negPadded.cutFromFront && (
                  <div className="text-xs text-amber-700 mt-1">⚠️ Overlength — kept last {blockSize} tokens (front truncated).</div>
                )}
              </div>
              <div>
                <Label>Mask (False = active, True = ignore)</Label>
                <MaskViz mask={negPadded.mask} />
                <div className="text-xs text-gray-500 mt-1">Following the code: indices &gt;= len(concat) + 1 are masked (when padding case).</div>
              </div>
            </div>
          </div>
        </div>

        {/* Final __getitem__ preview */}
        <div className="mt-6 border rounded-lg p-4 bg-white">
          <h3 className="font-semibold mb-2">What __getitem__ returns</h3>
          <pre className="text-xs bg-slate-900 text-slate-50 p-3 rounded overflow-auto">{JSON.stringify({
            pos_toks: posPadded.toks,
            pos_mask: posPadded.mask,
            neg_toks: negPadded.toks,
            neg_mask: negPadded.mask,
          }, null, 2)}</pre>
        </div>

        {/* Notes */}
        <div className="mt-6 text-sm text-gray-700 space-y-2">
          <div><span className="font-semibold">Filter step:</span> We drop an item if either concatenation would exceed <span className="font-mono">block_size + 1</span> (mirrors the Python condition).
          </div>
          <div><span className="font-semibold">Padding rule:</span> If the concatenation is shorter than <span className="font-mono">block_size</span>, we pad with EOT (<span className="font-mono">{EOT_ID}</span>). The mask is set to <span className="font-mono">True</span> from index <span className="font-mono">len(concat)+1</span> onward, matching the snippet.
          </div>
          <div><span className="font-semibold">Truncation rule:</span> If it is longer/equal to <span className="font-mono">block_size</span>, we keep the <em>last</em> <span className="font-mono">block_size</span> tokens (front-truncation).</div>
        </div>
      </div>
    </div>
  );
}
