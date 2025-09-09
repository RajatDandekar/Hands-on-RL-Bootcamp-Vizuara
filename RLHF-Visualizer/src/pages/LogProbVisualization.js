import React, { useState } from 'react';
import { ChevronRight, Play, RotateCcw, Calculator, ArrowDown, Zap, Brain, Hash } from 'lucide-react';

const LogProbVisualization = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedToken, setSelectedToken] = useState(null);
  const [showArchitecture, setShowArchitecture] = useState(false);
  const [archStep, setArchStep] = useState(0);

  // Example data
  const prompt = "Where is Pune?";
  const response = " Pune is in India.";
  const tokens = ["Where", " is", " Pune", "?", " Pune", " is", " in", " India", "."];
  const tokenIds = [2940, 318, 24723, 30, 24723, 318, 287, 3794, 13];
  
  // Simulated model outputs (logits before softmax)
  const logits = [
    [2.1, 8.5, 1.2, 0.8, 3.2], // For predicting " is"
    [1.5, 3.2, 9.1, 2.1, 1.8], // For predicting " Pune"
    [0.8, 2.1, 1.9, 7.8, 2.3], // For predicting "?"
    [3.1, 8.9, 2.4, 1.6, 2.8], // For predicting " Pune"
    [1.2, 2.8, 9.4, 1.9, 2.1], // For predicting " is"
    [2.3, 1.8, 3.1, 8.7, 1.9], // For predicting " in"
    [1.9, 2.4, 1.1, 2.8, 9.2], // For predicting " India"
    [8.1, 1.8, 2.3, 1.4, 2.9], // For predicting "."
  ];

  // Architecture steps for single token flow
  const archSteps = [
    { name: "Token Input", description: "Token 'Where' (ID: 2940) enters the model", active: false },
    { name: "Embedding", description: "Convert token ID to dense vector representation", active: false },
    { name: "Positional Encoding", description: "Add position information to embedding", active: false },
    { name: "Multi-Head Attention", description: "Attend to previous tokens (self-attention)", active: false },
    { name: "Add & Norm", description: "Residual connection + layer normalization", active: false },
    { name: "Feed Forward", description: "2-layer MLP with non-linear activation", active: false },
    { name: "Add & Norm", description: "Another residual connection + normalization", active: false },
    { name: "Output Projection", description: "Project to vocabulary size (50k+ dimensions)", active: false },
    { name: "Softmax", description: "Convert logits to probability distribution", active: false },
    { name: "Log Probability", description: "Take log of target token probability", active: false }
  ];

  // Vocabulary sample (simplified)
  const vocab = ["the", "is", "Pune", "?", "in", "India", ".", "a", "of"];
  const correctTokenIndices = [1, 2, 3, 2, 1, 4, 5, 6];

  const steps = [
    {
      title: "1. Token Pairs Setup",
      description: "We have input tokens and their corresponding target tokens"
    },
    {
      title: "2. Transformer Architecture",
      description: "See how a single token flows through the transformer to produce probabilities"
    },
    {
      title: "3. Model Forward Pass",
      description: "For each input token, the model outputs logits (raw scores) for all vocabulary tokens"
    },
    {
      title: "4. Softmax & Log Probabilities",
      description: "Convert logits to probabilities and extract log probabilities for target tokens"
    }
  ];

  const getInputTokens = () => tokens.slice(0, -1);
  const getTargetTokens = () => tokens.slice(1);

  const softmax = (logitsArray) => {
    const maxLogit = Math.max(...logitsArray);
    const expLogits = logitsArray.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
  };

  const getLogProb = (logitsArray, targetIndex) => {
    const probs = softmax(logitsArray);
    return Math.log(probs[targetIndex]);
  };

  const formatProb = (prob) => (prob * 100).toFixed(1) + '%';
  const formatLogProb = (logProb) => logProb.toFixed(3);

  const animateStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const reset = () => {
    setCurrentStep(0);
    setSelectedToken(null);
    setShowArchitecture(false);
    setArchStep(0);
  };

  const animateArch = () => {
    if (archStep < archSteps.length - 1) {
      setArchStep(archStep + 1);
    }
  };

  const resetArch = () => {
    setArchStep(0);
  };

  // Transformer Architecture Component
  const TransformerArchitecture = () => {
    const activeColor = (stepIndex) => archStep >= stepIndex ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-white';
    const textColor = (stepIndex) => archStep >= stepIndex ? 'text-blue-800' : 'text-gray-600';

    return (
      <div className="bg-gradient-to-b from-blue-50 to-white p-6 rounded-lg border-2 border-blue-200">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold text-gray-800">Transformer Architecture Flow</h3>
          <div className="space-x-2">
            <button
              onClick={animateArch}
              disabled={archStep >= archSteps.length - 1}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
            >
              Next Step
            </button>
            <button
              onClick={resetArch}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 text-sm"
            >
              Reset
            </button>
          </div>
        </div>

        <div className="flex flex-col items-center space-y-4">
          {/* Input Token */}
          <div className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${activeColor(0)}`}>
            <div className="text-center">
              <Hash className="w-6 h-6 mx-auto mb-2 text-blue-600" />
              <div className="font-bold">Token: "Where"</div>
              <div className="text-sm text-gray-600">ID: 2940</div>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-gray-400" />

          {/* Embedding Layer */}
          <div className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${activeColor(1)}`}>
            <div className="text-center">
              <div className="font-semibold">Token Embedding</div>
              <div className="text-xs mt-1">2940 → [0.1, -0.3, 0.8, ...]</div>
              <div className="text-xs text-gray-500">Shape: [1, 768]</div>
            </div>
          </div>

          <div className="text-lg font-bold text-gray-400">+</div>

          {/* Positional Encoding */}
          <div className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${activeColor(2)}`}>
            <div className="text-center">
              <div className="font-semibold">Positional Encoding</div>
              <div className="text-xs mt-1">pos=0 → [0.0, 1.0, 0.0, ...]</div>
              <div className="text-xs text-gray-500">Shape: [1, 768]</div>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-gray-400" />

          {/* Transformer Layers */}
          <div className={`w-64 p-4 rounded-lg border-2 transition-all duration-500 ${activeColor(3)} relative`}>
            <div className="text-center mb-3">
              <Brain className="w-6 h-6 mx-auto mb-2 text-purple-600" />
              <div className="font-semibold">Transformer Layers (×12)</div>
            </div>
            
            <div className="space-y-2 text-sm">
              <div className={`p-2 rounded border ${archStep >= 3 ? 'bg-purple-100 border-purple-300' : 'bg-gray-50'}`}>
                <div className="font-medium">Multi-Head Attention</div>
                <div className="text-xs text-gray-600">Query, Key, Value matrices</div>
              </div>
              
              <div className={`p-2 rounded border ${archStep >= 4 ? 'bg-purple-100 border-purple-300' : 'bg-gray-50'}`}>
                <div className="font-medium">Add & Norm</div>
                <div className="text-xs text-gray-600">Residual + LayerNorm</div>
              </div>
              
              <div className={`p-2 rounded border ${archStep >= 5 ? 'bg-purple-100 border-purple-300' : 'bg-gray-50'}`}>
                <div className="font-medium">Feed Forward Network</div>
                <div className="text-xs text-gray-600">768 → 3072 → 768</div>
              </div>
              
              <div className={`p-2 rounded border ${archStep >= 6 ? 'bg-purple-100 border-purple-300' : 'bg-gray-50'}`}>
                <div className="font-medium">Add & Norm</div>
                <div className="text-xs text-gray-600">Final residual connection</div>
              </div>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-gray-400" />

          {/* Output Projection */}
          <div className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${activeColor(7)}`}>
            <div className="text-center">
              <div className="font-semibold">Linear Projection</div>
              <div className="text-xs mt-1">[768] → [50,257]</div>
              <div className="text-xs text-gray-500">To vocabulary size</div>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-gray-400" />

          {/* Logits */}
          <div className={`w-64 p-3 rounded-lg border-2 transition-all duration-500 ${activeColor(7)}`}>
            <div className="text-center">
              <div className="font-semibold">Raw Logits</div>
              <div className="flex justify-center gap-1 mt-2 flex-wrap">
                {logits[0].map((logit, idx) => (
                  <div key={idx} className={`px-2 py-1 rounded text-xs ${
                    idx === 1 ? 'bg-green-200 font-bold' : 'bg-gray-100'
                  }`}>
                    {vocab[idx]}: {logit.toFixed(1)}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-gray-400" />

          {/* Softmax */}
          <div className={`w-48 p-3 rounded-lg border-2 transition-all duration-500 ${activeColor(8)}`}>
            <div className="text-center">
              <Zap className="w-6 h-6 mx-auto mb-2 text-yellow-600" />
              <div className="font-semibold">Softmax</div>
              <div className="text-xs mt-1">exp(logits) / Σexp(logits)</div>
            </div>
          </div>

          <ArrowDown className="w-5 h-5 text-gray-400" />

          {/* Final Probabilities */}
          <div className={`w-64 p-3 rounded-lg border-2 transition-all duration-500 ${activeColor(9)}`}>
            <div className="text-center">
              <div className="font-semibold">Probability Distribution</div>
              <div className="flex justify-center gap-1 mt-2 flex-wrap">
                {softmax(logits[0]).map((prob, idx) => (
                  <div key={idx} className={`px-2 py-1 rounded text-xs ${
                    idx === 1 ? 'bg-green-200 font-bold' : 'bg-gray-100'
                  }`}>
                    {vocab[idx]}: {formatProb(prob)}
                  </div>
                ))}
              </div>
              <div className="mt-2 p-2 bg-red-100 border border-red-300 rounded">
                <div className="text-sm font-bold">log(P(" is")) = {formatLogProb(getLogProb(logits[0], 1))}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Step Description */}
        {archStep < archSteps.length && (
          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h4 className="font-semibold text-yellow-800 mb-2">
              Current Step: {archSteps[archStep].name}
            </h4>
            <p className="text-yellow-700 text-sm">{archSteps[archStep].description}</p>
          </div>
        )}
      </div>
    );
  };

  const TokenProbabilityDetails = ({ tokenIndex }) => {
    if (tokenIndex === null) return null;
    
    const tokenLogits = logits[tokenIndex];
    const probs = softmax(tokenLogits);
    const correctIndex = correctTokenIndices[tokenIndex];
    const logProb = getLogProb(tokenLogits, correctIndex);

    return (
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-semibold text-blue-800 mb-3">
          Detailed Calculation for: "{getInputTokens()[tokenIndex]}" → "{getTargetTokens()[tokenIndex]}"
        </h4>
        
        <div className="space-y-4">
          <div>
            <h5 className="font-medium text-gray-700 mb-2">1. Raw Logits (model output):</h5>
            <div className="flex gap-2 flex-wrap">
              {tokenLogits.map((logit, idx) => (
                <div key={idx} className={`px-2 py-1 rounded text-sm ${
                  idx === correctIndex ? 'bg-green-200 border-2 border-green-400' : 'bg-gray-100'
                }`}>
                  {vocab[idx]}: {logit.toFixed(1)}
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ArrowDown className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600">Apply Softmax</span>
          </div>

          <div>
            <h5 className="font-medium text-gray-700 mb-2">2. Probabilities (after softmax):</h5>
            <div className="flex gap-2 flex-wrap">
              {probs.map((prob, idx) => (
                <div key={idx} className={`px-2 py-1 rounded text-sm ${
                  idx === correctIndex ? 'bg-green-200 border-2 border-green-400' : 'bg-gray-100'
                }`}>
                  {vocab[idx]}: {formatProb(prob)}
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ArrowDown className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600">Take log of target probability</span>
          </div>

          <div>
            <h5 className="font-medium text-gray-700 mb-2">3. Log Probability:</h5>
            <div className="bg-yellow-100 border border-yellow-300 rounded p-3">
              <div className="font-mono text-lg">
                log({formatProb(probs[correctIndex])}) = <span className="font-bold text-red-600">{formatLogProb(logProb)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center mb-2 text-gray-800">
          Log Probability Calculation in Transformers
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Understanding <code className="bg-gray-100 px-2 py-1 rounded">model.log_probs(completion_minus_1, target)</code> with transformer architecture
        </p>

        {/* Controls */}
        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={animateStep}
            disabled={currentStep >= steps.length - 1}
            className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Play className="w-4 h-4" />
            <span>Next Step</span>
          </button>
          <button
            onClick={reset}
            className="flex items-center space-x-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>

        {/* Step indicator */}
        <div className="mb-8">
          <div className="bg-blue-100 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-2 text-blue-800">
              {steps[currentStep].title}
            </h2>
            <p className="text-blue-700">{steps[currentStep].description}</p>
          </div>
        </div>

        {/* Code Context */}
        <div className="mb-8 bg-gray-900 text-white p-4 rounded-lg font-mono text-sm">
          <div className="text-green-400 mb-2"># Your code:</div>
          <div>completion = model.generate("Where is Pune?", ...)</div>
          <div className={`${currentStep >= 0 ? 'text-yellow-400' : 'text-gray-500'}`}>
            completion_minus_1, target = completion[:, :-1], completion[:, 1:]
          </div>
          <div className={`${currentStep >= 2 ? 'text-yellow-400' : 'text-gray-500'}`}>
            original_log_prob = model.log_probs(completion_minus_1, target)
          </div>
        </div>

        {/* Visualization */}
        <div className="space-y-8">
          {/* Step 1: Token Pairs */}
          {currentStep >= 0 && (
            <div className={`transition-all duration-500 ${currentStep === 0 ? 'ring-2 ring-blue-500' : ''} bg-gray-50 p-6 rounded-lg`}>
              <h3 className="text-lg font-semibold mb-4 text-gray-800">Input-Target Token Pairs:</h3>
              <div className="space-y-3">
                {getInputTokens().map((inputToken, idx) => (
                  <div key={idx} className="flex items-center justify-center space-x-4 p-3 bg-white rounded-lg border">
                    <div className="bg-blue-100 border border-blue-300 rounded-lg p-2 text-center min-w-[100px]">
                      <div className="font-mono text-xs bg-blue-200 px-1 rounded mb-1">
                        {tokenIds[idx]}
                      </div>
                      <div className="font-medium text-sm">"{inputToken}"</div>
                    </div>
                    
                    <ChevronRight className="w-5 h-5 text-gray-400" />
                    <span className="text-gray-500 font-medium">predicts</span>
                    <ChevronRight className="w-5 h-5 text-gray-400" />
                    
                    <div className="bg-green-100 border border-green-300 rounded-lg p-2 text-center min-w-[100px]">
                      <div className="font-mono text-xs bg-green-200 px-1 rounded mb-1">
                        {tokenIds[idx + 1]}
                      </div>
                      <div className="font-medium text-sm">"{getTargetTokens()[idx]}"</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Step 2: Transformer Architecture */}
          {currentStep >= 1 && (
            <div className={`transition-all duration-500 ${currentStep === 1 ? 'ring-2 ring-blue-500' : ''}`}>
              <TransformerArchitecture />
            </div>
          )}

          {/* Step 3: Model Forward Pass */}
          {currentStep >= 2 && (
            <div className={`transition-all duration-500 ${currentStep === 2 ? 'ring-2 ring-blue-500' : ''} bg-gray-50 p-6 rounded-lg`}>
              <h3 className="text-lg font-semibold mb-4 text-gray-800">Model Forward Pass - All Token Logits:</h3>
              <p className="text-gray-600 mb-4">The same process happens for each input token position:</p>
              
              <div className="space-y-3">
                {getInputTokens().map((inputToken, idx) => (
                  <div key={idx} className="bg-white border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">Input: "{inputToken}" (position {idx})</span>
                      <span className="text-sm text-gray-500">Target: "{getTargetTokens()[idx]}"</span>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      {logits[idx].map((logit, vocabIdx) => (
                        <div key={vocabIdx} className={`px-2 py-1 rounded text-xs ${
                          vocabIdx === correctTokenIndices[idx] 
                            ? 'bg-green-200 border-2 border-green-400 font-bold' 
                            : 'bg-gray-100'
                        }`}>
                          {vocab[vocabIdx]}: {logit.toFixed(1)}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Step 4: Softmax & Log Probabilities */}
          {currentStep >= 3 && (
            <div className={`transition-all duration-500 ${currentStep === 3 ? 'ring-2 ring-blue-500' : ''} bg-gray-50 p-6 rounded-lg`}>
              <h3 className="text-lg font-semibold mb-4 text-gray-800">Final Log Probabilities:</h3>
              <p className="text-gray-600 mb-4">Click on any row to see detailed calculation. Extract log probabilities for the actual target tokens:</p>
              
              <div className="space-y-3">
                {getInputTokens().map((inputToken, idx) => {
                  const logProb = getLogProb(logits[idx], correctTokenIndices[idx]);
                  const prob = softmax(logits[idx])[correctTokenIndices[idx]];
                  
                  return (
                    <div 
                      key={idx} 
                      className={`bg-white border rounded-lg p-4 cursor-pointer hover:bg-blue-50 transition-colors ${
                        selectedToken === idx ? 'ring-2 ring-blue-400' : ''
                      }`}
                      onClick={() => setSelectedToken(selectedToken === idx ? null : idx)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <span className="font-medium">"{inputToken}" → "{getTargetTokens()[idx]}"</span>
                          <div className="text-sm text-gray-500">
                            P = {formatProb(prob)}
                          </div>
                          <Calculator className="w-4 h-4 text-blue-500" />
                        </div>
                        <div className="bg-red-100 border border-red-300 rounded px-3 py-1">
                          <span className="font-mono font-bold text-red-700">
                            log_prob = {formatLogProb(logProb)}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              
              <TokenProbabilityDetails tokenIndex={selectedToken} />

              <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <h4 className="font-semibold text-yellow-800 mb-2">Key Insights:</h4>
                <ul className="text-sm text-yellow-700 space-y-1">
                  <li>• Each token position runs through the entire transformer architecture</li>
                  <li>• Attention layers let tokens "see" previous context</li>
                  <li>• Final linear layer projects to vocabulary space</li>
                  <li>• Log probabilities measure model confidence/surprise</li>
                  <li>• In RLHF, these original log probs are compared with new policy log probs</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* Mathematical Summary */}
        {currentStep >= 3 && (
          <div className="mt-8 p-6 bg-indigo-50 rounded-lg border border-indigo-200">
            <h3 className="text-lg font-semibold text-indigo-800 mb-3">Complete Mathematical Flow:</h3>
            <div className="bg-white p-4 rounded border font-mono text-sm space-y-2">
              <div><strong>1. Input Processing:</strong></div>
              <div className="ml-4">x = token_embedding + positional_encoding</div>
              
              <div><strong>2. Transformer Layers:</strong></div>
              <div className="ml-4">for layer in layers:</div>
              <div className="ml-8">attn_out = MultiHeadAttention(x)</div>
              <div className="ml-8">x = LayerNorm(x + attn_out)</div>
              <div className="ml-8">ffn_out = FeedForward(x)</div>
              <div className="ml-8">x = LayerNorm(x + ffn_out)</div>
              
              <div><strong>3. Output:</strong></div>
              <div className="ml-4">logits = LinearProjection(x)  # Shape: [seq_len, vocab_size]</div>
              <div className="ml-4">probs = softmax(logits)</div>
              <div className="ml-4">log_probs = log(probs[target_indices])</div>
              
              <div className="mt-4 text-indigo-700">
                <strong>Result: original_log_prob = [log_prob[0], log_prob[1], ..., log_prob[n-1]]</strong>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LogProbVisualization;