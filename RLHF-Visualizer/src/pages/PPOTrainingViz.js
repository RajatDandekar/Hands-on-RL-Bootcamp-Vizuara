import React, { useState, useEffect, useCallback, useMemo } from "react";

/**
 * Interactive PPO Training Visualization — v2
 * --------------------------------------------------
 * Implements requested changes:
 * 1) Prompts now include the full question (e.g., "Where is Pune?") and
 *    completions are concise answers (e.g., "Pune is in India.").
 * 2) Step 2 shows old log-probabilities for ALL tokens of one example
 *    (prompt tokens display as "—").
 * 3) Step 3 clearly ties GAE values (A_t, R_t) to individual tokens with
 *    explicit Pos/Token rows.
 * 4) Step 6 adds mini bar charts for PPO loss and Value loss (visuals),
 *    alongside the table; Step 8 includes a gradient/chain rule panel.
 */

// ---------------------------------
// Icons (self-contained SVGs)
// ---------------------------------
const Icon = ({ children, className = "w-6 h-6" }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    {children}
  </svg>
);
const Play = ({ className }) => <Icon className={className}><polygon points="5 3 19 12 5 21 5 3" /></Icon>;
const Pause = ({ className }) => <Icon className={className}><rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" /></Icon>;
const RotateCcw = ({ className }) => <Icon className={className}><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /></Icon>;
const ChevronRight = ({ className }) => <Icon className={className}><polyline points="9 18 15 12 9 6" /></Icon>;
const Settings = ({ className }) => (
  <Icon className={className}>
    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 0 2l-.15.08a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1 0-2l.15-.08a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15-.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
    <circle cx="12" cy="12" r="3" />
  </Icon>
);
const BarChart3 = ({ className }) => <Icon className={className}><path d="M3 3v18h18" /><path d="M18 17V9" /><path d="M13 17V5" /><path d="M8 17v-3" /></Icon>;

// ---------------------------------
// Demo Data Synthesis
// ---------------------------------
const getInitialSequences = () => {
  const createTokenData = (position, token, isPrompt) => ({
    position,
    token,
    isPrompt,
    isGenerated: !isPrompt,
    oldLogProb: !isPrompt ? -Math.random() * 2 - 0.5 : null,
    advantage: !isPrompt ? (Math.random() - 0.5) * 1.5 : null,
    return: !isPrompt ? Math.random() * 2.5 + 1 : null,
    targetValue: !isPrompt ? Math.random() * 2 + 1 : null,
    newLogProb: null,
    currentValue: null,
    ratio: null,
    unclippedObj: null,
    clippedObj: null,
    ppoLoss: null,
    valueSqErr: null,
    entropy: null,
    totalTokenLoss: null,
  });

  // CHANGE 1: prompts contain full question; completions are concise answers
  const baseSequences = [
    { id: 0, prompt: ["Where", " is", " Pune", "?"],       completion: [" Pune", " is", " in", " India", "."] },
    { id: 1, prompt: ["Where", " is", " Mumbai", "?"],     completion: [" Mumbai", " is", " in", " India", "."] },
    { id: 2, prompt: ["Where", " is", " Delhi", "?"],      completion: [" Delhi", " is", " in", " India", "."] },
    { id: 3, prompt: ["What",  " is", " Pune", "?"],       completion: [" Pune", " is", " a", " city", "."] },
    { id: 4, prompt: ["Where", " is", " Chennai", "?"],    completion: [" Chennai", " is", " in", " India", "."] },
    { id: 5, prompt: ["Where", " is", " Kolkata", "?"],    completion: [" Kolkata", " is", " in", " India", "."] },
  ];

  return baseSequences.map((seq) => {
    const allTokens = [...seq.prompt, ...seq.completion];
    const tokens = allTokens.map((token, idx) => createTokenData(idx, token, idx < seq.prompt.length));
    return { ...seq, tokens, generatedTokens: tokens.filter((t) => t.isGenerated) };
  });
};

// ---------------------------------
// Helper: PPO math
// ---------------------------------
function computePPOForToken(t, eps, valueCoef, entropyCoef, useValueClip, valueClipRange) {
  if (!t.isGenerated) return t;
  const ratio = t.newLogProb != null && t.oldLogProb != null ? Math.exp(t.newLogProb - t.oldLogProb) : null;
  const A = t.advantage ?? 0;
  const unclippedObj = ratio != null ? ratio * A : null;
  const clippedRatio = ratio != null ? Math.max(1 - eps, Math.min(1 + eps, ratio)) : null;
  const clippedObj = clippedRatio != null ? clippedRatio * A : null;
  const ppoLoss = unclippedObj != null && clippedObj != null ? -Math.min(unclippedObj, clippedObj) : null;

  const vPred = t.currentValue ?? 0;
  const R = t.return ?? 0;
  let valueSqErr;
  if (useValueClip && t.targetValue != null) {
    const vOld = t.targetValue;
    const vClipped = Math.max(vOld - valueClipRange, Math.min(vOld + valueClipRange, vPred));
    const err1 = (vPred - R) ** 2;
    const err2 = (vClipped - R) ** 2;
    valueSqErr = Math.max(err1, err2);
  } else {
    valueSqErr = (vPred - R) ** 2;
  }
  const entropy = t.newLogProb != null ? -t.newLogProb : 0;

  return {
    ...t,
    ratio,
    unclippedObj,
    clippedObj,
    ppoLoss,
    valueSqErr,
    entropy,
    totalTokenLoss: (ppoLoss ?? 0) + valueCoef * (valueSqErr ?? 0) - entropyCoef * (entropy ?? 0),
  };
}

const PPOTrainingViz = () => {
  // Playback & step state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentUpdate, setCurrentUpdate] = useState(0);
  const [currentMiniBatch, setCurrentMiniBatch] = useState(0);
  const [currentTokenIndex, setCurrentTokenIndex] = useState(0);
  const [animationSpeed, setAnimationSpeed] = useState(2000);
  const [showDetails, setShowDetails] = useState(false);
  const [showFormulas, setShowFormulas] = useState(true);

  // PPO hyperparams (interactive)
  const [clipEps, setClipEps] = useState(0.2);
  const [valueCoef, setValueCoef] = useState(0.5);
  const [entropyCoef, setEntropyCoef] = useState(0.01);
  const [useValueClip, setUseValueClip] = useState(true);
  const [valueClipRange, setValueClipRange] = useState(0.2);

  // Data
  const [sequencesData, setSequencesData] = useState(getInitialSequences);

  // Batching config
  const sample_batch_size = 6;
  const train_batch_size = 2;
  const n_updates = 2;

  // Global permutation for shuffling
  const [currentPermutation, setCurrentPermutation] = useState(() => Array.from({ length: sample_batch_size }, (_, i) => i));

  const steps = [
    { title: "Data Collection", description: "Sequences with token-level metrics are gathered from the model." },
    { title: "Old Log Probs Calculation", description: "Compute log-probabilities under the old policy for all tokens of one example (prompts show —)." },
    { title: "GAE Calculation", description: "Show A_t and R_t for each generated token with clear Pos/Token mapping." },
    { title: "Sequence Shuffle", description: "Randomly permute the batch before creating mini-batches." },
    { title: "Mini-Batch Split", description: "Split the permuted batch into mini-batches for multiple SGD epochs." },
    { title: "Forward Pass", description: "Run the new policy to get NEW log-probs and value predictions for tokens in this mini-batch." },
    { title: "PPO & Value Loss", description: "Compute per-token losses and visualize them with bar charts and a table." },
    { title: "Backward Pass", description: "Chain rule through policy/value heads; show gradient directions and formula." },
  ];

  const getMiniBatchByIndex = useCallback((permutation, sequences, miniBatchIndex) => {
    const start = miniBatchIndex * train_batch_size;
    const end = Math.min(start + train_batch_size, sample_batch_size);
    if (start >= permutation.length) return [];
    return permutation.slice(start, end).map((idx) => sequences[idx]);
  }, []);

  // Step driver
  const nextStep = useCallback(() => {
    const miniBatch = getMiniBatchByIndex(currentPermutation, sequencesData, currentMiniBatch);
    const totalTokensInMiniBatch = miniBatch.reduce((sum, seq) => sum + (seq.generatedTokens?.length || 0), 0);
    const isTokenLevelStep = currentStep >= 5 && currentStep <= 6;

    if (isTokenLevelStep && totalTokensInMiniBatch > 0 && currentTokenIndex < totalTokensInMiniBatch - 1) {
      setCurrentTokenIndex((prev) => prev + 1);
      return;
    }

    let nextStepVal = currentStep + 1;
    let nextMiniBatchVal = currentMiniBatch;
    let nextUpdateVal = currentUpdate;
    let nextPermutation = currentPermutation;

    if (nextStepVal >= steps.length) {
      const totalMiniBatches = Math.ceil(sample_batch_size / train_batch_size);
      if (currentMiniBatch < totalMiniBatches - 1) {
        nextStepVal = 5;
        nextMiniBatchVal = currentMiniBatch + 1;
      } else if (currentUpdate < n_updates - 1) {
        nextStepVal = 3;
        nextMiniBatchVal = 0;
        nextUpdateVal = currentUpdate + 1;
        const newPerm = [...Array(sample_batch_size)].map((_, i) => i);
        for (let i = newPerm.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [newPerm[i], newPerm[j]] = [newPerm[j], newPerm[i]];
        }
        nextPermutation = newPerm;
      } else {
        setIsPlaying(false);
        return;
      }
    }

    setCurrentStep(nextStepVal);
    setCurrentTokenIndex(0);
    setCurrentMiniBatch(nextMiniBatchVal);
    setCurrentUpdate(nextUpdateVal);
    setCurrentPermutation(nextPermutation);
  }, [currentStep, currentTokenIndex, currentMiniBatch, currentUpdate, sequencesData, currentPermutation, getMiniBatchByIndex]);

  const resetAnimation = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setCurrentUpdate(0);
    setCurrentMiniBatch(0);
    setCurrentTokenIndex(0);
    setSequencesData(getInitialSequences());
    setCurrentPermutation(Array.from({ length: sample_batch_size }, (_, i) => i));
  };

  useEffect(() => {
    if (isPlaying) {
      const timer = setTimeout(nextStep, animationSpeed);
      return () => clearTimeout(timer);
    }
  }, [isPlaying, nextStep, animationSpeed]);

  // Forward pass: populate new log-probs & values; reshuffle clears them
  useEffect(() => {
    if (currentStep === 5) {
      setSequencesData((prevData) => {
        const miniBatch = getMiniBatchByIndex(currentPermutation, prevData, currentMiniBatch);
        const firstToken = miniBatch.length > 0 && miniBatch[0].generatedTokens.length > 0 ? miniBatch[0].generatedTokens[0] : null;
        if (firstToken && firstToken.newLogProb !== null) return prevData;
        return prevData.map((seq) => {
          const isInBatch = miniBatch.some((bSeq) => bSeq.id === seq.id);
          if (!isInBatch) return seq;
          return {
            ...seq,
            tokens: seq.tokens.map((token) => token.isGenerated
              ? {
                  ...token,
                  newLogProb: (token.oldLogProb || 0) + (Math.random() - 0.5) * 0.4,
                  currentValue: (token.targetValue || 0) + (Math.random() - 0.5) * 0.5,
                }
              : token
            ),
          };
        });
      });
    } else if (currentStep === 3) {
      setSequencesData((prevData) => prevData.map((seq) => ({
        ...seq,
        tokens: seq.tokens.map((token) => ({
          ...token,
          newLogProb: null,
          currentValue: null,
          ratio: null,
          unclippedObj: null,
          clippedObj: null,
          ppoLoss: null,
          valueSqErr: null,
          entropy: null,
          totalTokenLoss: null,
        })),
      })));
    }
  }, [currentStep, currentMiniBatch, currentUpdate, currentPermutation, getMiniBatchByIndex]);

  // Recompute losses when at Step 6 or when hyperparams change
  useEffect(() => {
    if (currentStep !== 6) return;
    setSequencesData((prev) => prev.map((seq) => ({
      ...seq,
      tokens: seq.tokens.map((t) => (t.isGenerated && t.newLogProb != null)
        ? computePPOForToken(t, clipEps, valueCoef, entropyCoef, useValueClip, valueClipRange)
        : t
      ),
    })));
  }, [currentStep, clipEps, valueCoef, entropyCoef, useValueClip, valueClipRange]);

  // ------------------ UI Bits ------------------
  const TokenVisualization = ({ token, isHighlighted, showMetrics, compact }) => {
    const bgColor = token.isPrompt ? "bg-blue-100 border-blue-300" : "bg-green-100 border-green-300";
    const highlightColor = isHighlighted ? "ring-2 ring-orange-400 bg-orange-200 scale-105 shadow-lg" : "";
    return (
      <div className={`border-2 rounded ${compact ? "p-1" : "p-2"} text-center transition-all duration-300 ${bgColor} ${highlightColor}`}>
        <div className={`font-mono font-bold ${compact ? "text-xs" : "text-sm"}`}>"{token.token}"</div>
        {showMetrics && token.isGenerated && (
          <div className="text-xs space-y-1 mt-1">
            <div className="text-red-600">old: {token.oldLogProb?.toFixed(2)}</div>
            <div className="text-green-600">new: {token.newLogProb?.toFixed(2)}</div>
            {token.ratio != null && <div className="text-blue-700">r: {token.ratio.toFixed(3)}</div>}
          </div>
        )}
      </div>
    );
  };

  const SequenceCard = ({ seq, showMetrics, activeTokenPosition, compact }) => (
    <div className={`border rounded-lg p-3 transition-all duration-300 bg-white`}>
      <div className="font-semibold mb-2">Sequence #{seq.id}</div>
      <div className="flex flex-wrap gap-1">
        {seq.tokens.map((token) => (
          <TokenVisualization key={token.position} token={token} isHighlighted={token.position === activeTokenPosition} showMetrics={showMetrics} compact={compact} />
        ))}
      </div>
      {showMetrics && seq.tokens.some((t) => t.totalTokenLoss != null) && (
        <div className="mt-3 text-xs grid grid-cols-2 gap-2">
          <div className="p-2 bg-slate-50 rounded border">
            <span className="font-semibold">Seq token losses (sum): </span>
            {seq.tokens.filter((t) => t.totalTokenLoss != null).reduce((acc, t) => acc + t.totalTokenLoss, 0).toFixed(3)}
          </div>
          <div className="p-2 bg-slate-50 rounded border">
            <span className="font-semibold">Value MSE sum: </span>
            {seq.tokens.filter((t) => t.valueSqErr != null).reduce((acc, t) => acc + t.valueSqErr, 0).toFixed(3)}
          </div>
        </div>
      )}
    </div>
  );

  const ControlPanel = () => (
    <div className="flex flex-wrap items-center justify-center gap-4 mb-6 p-4 bg-gray-100 rounded-lg">
      <button onClick={() => setIsPlaying(!isPlaying)} className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-white ${isPlaying ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"}`}>
        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        <span>{isPlaying ? "Pause" : "Play"}</span>
      </button>
      <button onClick={nextStep} disabled={isPlaying} className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
        <ChevronRight className="w-4 h-4" />
        <span>Step</span>
      </button>
      <button onClick={resetAnimation} className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700">
        <RotateCcw className="w-4 h-4" />
        <span>Reset</span>
      </button>
      <div className="hidden md:flex items-center space-x-2">
        <Settings className="w-4 h-4 text-gray-600" />
        <label className="text-sm">Speed:</label>
        <input type="range" min="500" max="4000" step="250" value={animationSpeed} onChange={(e) => setAnimationSpeed(Number(e.target.value))} className="w-28" />
      </div>
      <button onClick={() => setShowDetails(!showDetails)} className="flex items-center space-x-2 px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
        <BarChart3 className="w-4 h-4" />
        <span>{showDetails ? "Hide" : "Show"} Details</span>
      </button>
      <button onClick={() => setShowFormulas((v) => !v)} className="flex items-center space-x-2 px-3 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700">
        <span>Toggle Formulas</span>
      </button>
    </div>
  );

  const StepIndicator = () => (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">Update {currentUpdate + 1}/{n_updates} — Step {currentStep + 1}: {steps[currentStep].title}</h2>
        <div className="text-sm text-gray-600">Mini-batch {currentMiniBatch + 1}/{Math.ceil(sample_batch_size / train_batch_size)}</div>
      </div>
      <div className="bg-blue-100 rounded-lg p-4 mb-4 min-h-[60px]"><p className="text-blue-800">{steps[currentStep].description}</p></div>
      <div className="flex space-x-2 mb-4">{steps.map((_, idx) => (<div key={idx} className={`flex-1 h-3 rounded ${idx <= currentStep ? "bg-blue-500" : "bg-gray-300"}`} />))}</div>
    </div>
  );

  // CHANGE 4: Simple mini bar charts for losses
  const LossMiniCharts = () => {
    const miniBatchData = getMiniBatchByIndex(currentPermutation, sequencesData, currentMiniBatch);
    const tokens = [];
    miniBatchData.forEach((seq) => seq.tokens.forEach((t) => { if (t.isGenerated && t.totalTokenLoss != null) tokens.push(t); }));
    if (tokens.length === 0) return null;

    const barW = 16, gap = 6, height = 120, pad = 10;
    const ppoVals = tokens.map((t) => Math.max(0, t.ppoLoss ?? 0));
    const vVals = tokens.map((t) => Math.max(0, t.valueSqErr ?? 0));
    const maxP = Math.max(...ppoVals, 1e-6);
    const maxV = Math.max(...vVals, 1e-6);

    return (
      <div className="grid md:grid-cols-2 gap-4">
        <div className="border rounded p-3">
          <div className="text-sm font-semibold mb-1">PPO loss per token — lower is better</div>
          <svg width={(barW + gap) * ppoVals.length + pad} height={height}>
            {ppoVals.map((v, i) => {
              const h = (v / maxP) * (height - 2 * pad);
              return <rect key={i} x={pad + i * (barW + gap)} y={height - pad - h} width={barW} height={h} rx={3} />;
            })}
          </svg>
        </div>
        <div className="border rounded p-3">
          <div className="text-sm font-semibold mb-1">Value loss (MSE) per token — lower is better</div>
          <svg width={(barW + gap) * vVals.length + pad} height={height}>
            {vVals.map((v, i) => {
              const h = (v / maxV) * (height - 2 * pad);
              return <rect key={i} x={pad + i * (barW + gap)} y={height - pad - h} width={barW} height={h} rx={3} />;
            })}
          </svg>
        </div>
      </div>
    );
  };

  // CHANGE 4: Backward pass explanatory panel with formulas
  const BackwardPanel = () => (
    <div className="rounded-lg border bg-white p-4">
      <h3 className="text-lg font-semibold mb-2">Backward Pass — How gradients flow</h3>
      <div className="font-mono text-sm space-y-2">
        <div>r_t = exp( logπ_new(a_t|s_t) − logπ_old(a_t|s_t) )</div>
        <div>L_t^policy = −min( r_t · A_t, clip(r_t,1−ε,1+ε) · A_t )</div>
        <div>L_t^value  = ( V_θ(s_t) − R_t )^2</div>
        <div>L_t^entropy = −H(π_θ(·|s_t))</div>
        <div><span className="font-semibold">Total:</span> L = Σ_t [ L_t^policy + c1·L_t^value + (−c2)·L_t^entropy ]</div>
      </div>
      <ul className="list-disc ml-6 mt-3 text-sm text-gray-700">
        <li><span className="font-semibold">Policy head:</span> gradients push logπ_new(a_t|s_t) ↑ when A_t &gt; 0 (until clipped), ↓ when A_t &lt; 0.</li>
        <li><span className="font-semibold">Value head:</span> gradients regress V_θ(s_t) → R_t (MSE).</li>
        <li><span className="font-semibold">Entropy bonus:</span> encourages exploration by increasing output entropy.</li>
      </ul>
      <div className="mt-3 border rounded p-3 bg-slate-50 text-xs">
        tokens → transformer → <span className="text-orange-700">policy head (logits → logπ)</span> & <span className="text-blue-700">value head (V_θ)</span> → losses → sum → ∂/∂θ
      </div>
    </div>
  );

  // Aggregate losses table for the visible mini-batch
  const MiniBatchLossTable = () => {
    const miniBatchData = getMiniBatchByIndex(currentPermutation, sequencesData, currentMiniBatch);
    const rows = [];
    miniBatchData.forEach((seq) => seq.tokens.forEach((t) => {
      if (!t.isGenerated || t.totalTokenLoss == null) return;
      rows.push({ seqId: seq.id, pos: t.position, token: t.token, ratio: t.ratio, uncl: t.unclippedObj, clip: t.clippedObj, ppo: t.ppoLoss, vErr: t.valueSqErr, ent: t.entropy, total: t.totalTokenLoss });
    }));

    if (rows.length === 0) return null;

    const totals = rows.reduce((acc, r) => ({
      ppo: acc.ppo + (r.ppo ?? 0), v: acc.v + (r.vErr ?? 0), h: acc.h + (r.ent ?? 0), total: acc.total + (r.total ?? 0)
    }), { ppo: 0, v: 0, h: 0, total: 0 });

    return (
      <div className="rounded-lg border bg-white p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold">Mini-batch Loss Breakdown</h3>
          <div className="text-sm text-gray-600">(Aggregates reflect current hyperparameters)</div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs md:text-sm">
            <thead>
              <tr className="bg-slate-100 text-left">
                <th className="p-2">Seq</th><th className="p-2">Pos</th><th className="p-2">Token</th>
                <th className="p-2">r_t</th><th className="p-2">r*A</th><th className="p-2">clip(r)*A</th>
                <th className="p-2">PPO loss</th><th className="p-2">(V−R)^2</th><th className="p-2">Entropy≈−logπ</th><th className="p-2">Total</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={i % 2 ? "bg-slate-50" : ""}>
                  <td className="p-2">#{r.seqId}</td>
                  <td className="p-2">{r.pos}</td>
                  <td className="p-2 font-mono">{r.token}</td>
                  <td className="p-2">{r.ratio?.toFixed(3)}</td>
                  <td className="p-2">{r.uncl?.toFixed(3)}</td>
                  <td className="p-2">{r.clip?.toFixed(3)}</td>
                  <td className="p-2">{r.ppo?.toFixed(3)}</td>
                  <td className="p-2">{r.vErr?.toFixed(3)}</td>
                  <td className="p-2">{r.ent?.toFixed(3)}</td>
                  <td className="p-2 font-semibold">{r.total?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-slate-200 font-semibold">
                <td className="p-2" colSpan={6}>Totals</td>
                <td className="p-2">{totals.ppo.toFixed(3)}</td>
                <td className="p-2">{totals.v.toFixed(3)}</td>
                <td className="p-2">{totals.h.toFixed(3)}</td>
                <td className="p-2">{totals.total.toFixed(3)}</td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    );
  };

  const MainVisualization = () => {
    const miniBatchData = getMiniBatchByIndex(currentPermutation, sequencesData, currentMiniBatch);

    // Active token highlight for Steps 6–7
    const activeTokenPosition = useMemo(() => {
      if (currentStep < 5) return -1;
      let globalTokenCount = 0;
      for (const seq of miniBatchData) {
        const n = seq.generatedTokens?.length || 0;
        const rel = currentTokenIndex - globalTokenCount;
        if (rel >= 0 && rel < n) return seq.generatedTokens[rel].position;
        globalTokenCount += n;
      }
      return -1;
    }, [currentStep, currentTokenIndex, miniBatchData]);

    return (
      <div className="space-y-6">
        {/* Step 1 */}
        <div style={{ display: currentStep === 0 ? 'block' : 'none' }} className={`p-6 rounded-lg border-2 border-blue-400 bg-blue-50`}>
          <h3 className="text-lg font-bold mb-4 text-blue-800">Step 1: Data Collection</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {sequencesData.slice(0, 4).map((seq) => (<SequenceCard key={seq.id} seq={seq} />))}
          </div>
        </div>

        {/* Step 2: Old log-probs for ALL tokens (example seq 0) */}
        <div style={{ display: currentStep === 1 ? 'block' : 'none' }} className={`p-6 rounded-lg border-2 border-teal-400 bg-teal-50`}>
          <h3 className="text-lg font-bold mb-4 text-teal-800">Step 2: Old Log Probs Calculation</h3>
          {sequencesData.slice(0, 1).map((seq) => (
            <div key={seq.id} className="bg-white p-4 rounded border">
              <div className="font-mono bg-gray-100 p-3 rounded text-center mb-3">All tokens in Sequence #{seq.id} (prompt rows show —)</div>
              <div className="overflow-x-auto">
                <table className="min-w-full text-xs md:text-sm">
                  <thead>
                    <tr className="bg-slate-100 text-left"><th className="p-2">Pos</th><th className="p-2">Token</th><th className="p-2">Type</th><th className="p-2">old log P(token)</th></tr>
                  </thead>
                  <tbody>
                    {seq.tokens.map((t) => (
                      <tr key={t.position} className={t.position % 2 ? 'bg-slate-50' : ''}>
                        <td className="p-2">{t.position}</td>
                        <td className="p-2 font-mono">"{t.token}"</td>
                        <td className="p-2">{t.isPrompt ? 'prompt' : 'generated'}</td>
                        <td className="p-2 text-teal-700">{t.isPrompt ? '—' : t.oldLogProb?.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>

        {/* Step 3: GAE per-token with explicit mapping */}
        <div style={{ display: currentStep === 2 ? 'block' : 'none' }} className={`p-6 rounded-lg border-2 border-yellow-400 bg-yellow-50`}>
          <h3 className="text-lg font-bold mb-4 text-yellow-800">Step 3: GAE Calculation</h3>
          {sequencesData.slice(0, 1).map((seq) => (
            <div key={seq.id} className="bg-white p-4 rounded border">
              <div className="overflow-x-auto">
                <table className="min-w-full text-xs md:text-sm">
                  <thead>
                    <tr className="bg-slate-100 text-left"><th className="p-2">Pos</th><th className="p-2">Token</th><th className="p-2">A_t (Advantage)</th><th className="p-2">R_t (Return)</th></tr>
                  </thead>
                  <tbody>
                    {seq.generatedTokens.map((t) => (
                      <tr key={t.position} className={t.position % 2 ? 'bg-slate-50' : ''}>
                        <td className="p-2">{t.position}</td>
                        <td className="p-2 font-mono">"{t.token}"</td>
                        <td className="p-2 text-blue-700">{t.advantage?.toFixed(3)}</td>
                        <td className="p-2 text-orange-700">{t.return?.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-2 text-xs text-gray-600">Each row corresponds to a specific generated token t, so A_t and R_t are tied to that token.</div>
            </div>
          ))}
        </div>

        {/* Step 4 & 5 */}
        <div className="grid md:grid-cols-2 gap-6" style={{ display: currentStep === 3 || currentStep === 4 ? 'grid' : 'none' }}>
          <div className={`p-6 rounded-lg border-2 ${currentStep === 3 ? 'border-purple-400 bg-purple-50' : 'border-gray-300'}`}>
            <h3 className="text-lg font-bold mb-4 text-purple-800">Step 4: Sequence Permutation</h3>
            <div className="grid grid-cols-3 gap-2">{currentPermutation.map((originalIdx, newPos) => (<div key={newPos} className="text-center"><div className="bg-purple-100 border border-purple-300 rounded p-2 text-xs">Seq #{originalIdx}</div></div>))}</div>
          </div>
          <div className={`p-6 rounded-lg border-2 ${currentStep === 4 ? 'border-green-400 bg-green-50' : 'border-gray-300'}`}>
            <h3 className="text-lg font-bold mb-4 text-green-800">Step 5: Mini-Batch Split</h3>
            <div className="space-y-3">{miniBatchData.map((seq) => (<SequenceCard key={seq.id} seq={seq} compact />))}</div>
          </div>
        </div>

        {/* Step 6–8 */}
        <div style={{ display: currentStep >= 5 ? 'block' : 'none' }}>
          <div className={`p-6 rounded-lg border-2 ${currentStep === 5 ? 'border-cyan-400 bg-cyan-50' : currentStep === 6 ? 'border-orange-400 bg-orange-50' : 'border-red-400 bg-red-50'}`}>
            <h3 className="text-lg font-bold mb-4">{steps[currentStep].title}</h3>
            {showFormulas && currentStep >= 6 && (
              <div className="mb-4 rounded-lg border bg-white p-4">
                <h3 className="text-lg font-semibold mb-2">PPO Training Objective</h3>
                <div className="space-y-2 font-mono text-sm overflow-x-auto">
                  <div>r_t = exp( logπ_new(a_t|s_t) − logπ_old(a_t|s_t) )</div>
                  <div>L^CLIP = E_t [ min( r_t·A_t, clip(r_t,1−ε,1+ε)·A_t ) ]</div>
                  <div>L^V = E_t [ ( V_θ(s_t) − R_t )^2 ] <span className="text-gray-500">(optional value clipping)</span></div>
                  <div>L^H = E_t [ H(π_θ(·|s_t)) ]</div>
                  <div>Total (minimize): L = −L^CLIP + c1·L^V − c2·L^H</div>
                </div>
              </div>
            )}

            {currentStep === 6 && (
              <div className="space-y-4">
                <LossMiniCharts />
                <MiniBatchLossTable />
              </div>
            )}

            {currentStep === 7 && (
              <div className="mt-4"><BackwardPanel /></div>
            )}

            <div className="space-y-3 mt-4">{miniBatchData.map((seq) => (<SequenceCard key={seq.id} seq={seq} showMetrics activeTokenPosition={activeTokenPosition} />))}</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-4 sm:p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-4 sm:p-8">
        <h1 className="text-3xl sm:text-4xl font-bold text-center mb-4 text-gray-800">Interactive PPO Training Visualization</h1>
        <p className="text-center text-gray-600 mb-8">A step-by-step breakdown of Proximal Policy Optimization with token-level math, visuals, and gradient flow.</p>
        <ControlPanel />
        <StepIndicator />
        <MainVisualization />
        {showDetails && (
          <div className="mt-8 text-sm text-gray-700 space-y-2">
            <p><span className="font-semibold">About the numbers:</span> Synthetic token metrics are used for clarity. Adjust ε, c1, c2 to see how losses change.</p>
            <p><span className="font-semibold">Tip:</span> Advance to Step 6 for charts + table; Step 7 explains the backward pass.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PPOTrainingViz;
