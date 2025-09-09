import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, Calculator } from 'lucide-react';

const AdvantageCalculationViz = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [calculationStep, setCalculationStep] = useState(0);
  const [animatedToken, setAnimatedToken] = useState(-1);

  // Complete sequence: "Where is Pune? Pune is in India."
  const completion = ["Where", " is", " Pune", "?", " Pune", " is", " in", " India", "."];
  const tokens = completion.slice(0, -1); // All tokens except the last one for log prob calculation
  
  // Simulated data - expanded to match full completion length
  const originalLogProbs = [-1.2, -0.8, -2.1, -1.5, -0.9, -0.7, -1.8, -1.1];
  const refLogProbs = [-1.0, -0.9, -2.0, -1.4, -0.8, -0.8, -1.7, -1.0];
  const values = [0.1, 0.3, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]; // Values for all positions including final
  const reward = 2.5; // Final reward
  const klBeta = 0.1;
  const gamma = 0.99;
  const lambda = 0.95;

  // Calculate KL divergence for all tokens except the last one
  const kl = originalLogProbs.map((orig, i) => orig - refLogProbs[i]);
  
  // Calculate scores (negative KL for each token + final reward)
  const klRewards = kl.map(k => -klBeta * k);
  const scores = [...klRewards, reward];

  // Manual GAE calculation for visualization
  const calculateGAE = () => {
    let lastgaelam = 0;
    const advantages = new Array(scores.length).fill(0);
    
    for (let t = scores.length - 1; t >= 0; t--) {
      const nextValue = t < scores.length - 1 ? values[t + 1] : 0.0;
      const delta = scores[t] + gamma * nextValue - values[t];
      advantages[t] = lastgaelam = delta + gamma * lambda * lastgaelam;
    }
    
    const returns = advantages.map((adv, i) => adv + values[i]);
    return { advantages, returns };
  };

  const { advantages, returns } = calculateGAE();

  const steps = [
    {
      title: "1. Log Probabilities",
      description: "Compare original model log probs with reference model log probs for all tokens"
    },
    {
      title: "2. KL Divergence & Scores",
      description: "Calculate KL penalty and combine with reward to get scores"
    },
    {
      title: "3. Value Predictions",
      description: "Get value estimates for each token position from the value model"
    },
    {
      title: "4. GAE Advantage Calculation",
      description: "Calculate advantages using Generalized Advantage Estimation for all positions"
    }
  ];

  const animateStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      setCalculationStep(0);
      setAnimatedToken(-1);
    }
  };

  const reset = () => {
    setCurrentStep(0);
    setCalculationStep(0);
    setAnimatedToken(-1);
  };

  // Auto-animate GAE calculation
  useEffect(() => {
    if (currentStep === 3) {
      const interval = setInterval(() => {
        setCalculationStep(prev => {
          const next = (prev + 1) % (scores.length + 1);
          setAnimatedToken(next < scores.length ? scores.length - 1 - next : -1);
          return next;
        });
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [currentStep, scores.length]);

  // Log Probabilities Visualization
  const LogProbsVisualization = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-blue-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-blue-800">Log Probability Comparison</h3>
      
      <div className="grid md:grid-cols-2 gap-4">
        {tokens.map((token, i) => (
          <div key={i} className="bg-gray-50 p-4 rounded-lg border">
            <div className="flex items-center justify-between mb-3">
              <div className="font-medium">Token: "{token}"</div>
              <div className="text-sm text-gray-500">Position {i}</div>
            </div>
            
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="bg-blue-100 border border-blue-300 rounded p-2">
                  <div className="text-xs font-medium text-blue-700">Original</div>
                  <div className="text-sm font-bold text-blue-800">{originalLogProbs[i].toFixed(2)}</div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="bg-green-100 border border-green-300 rounded p-2">
                  <div className="text-xs font-medium text-green-700">Reference</div>
                  <div className="text-sm font-bold text-green-800">{refLogProbs[i].toFixed(2)}</div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="bg-red-100 border border-red-300 rounded p-2">
                  <div className="text-xs font-medium text-red-700">KL</div>
                  <div className="text-sm font-bold text-red-800">{kl[i].toFixed(2)}</div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-300 rounded">
        <div className="text-sm text-yellow-800">
          <strong>Formula:</strong> KL[i] = original_log_prob[i] - ref_log_prob[i]
        </div>
      </div>
    </div>
  );

  // Scores Visualization
  const ScoresVisualization = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-purple-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-purple-800">Score Calculation</h3>
      
      <div className="space-y-4">
        {/* KL Rewards */}
        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
          <h4 className="font-semibold mb-3 text-purple-700">KL Penalty Rewards</h4>
          <div className="grid grid-cols-4 gap-2">
            {klRewards.map((klReward, i) => (
              <div key={i} className="text-center">
                <div className="bg-white border-2 border-purple-300 rounded-lg p-2">
                  <div className="text-xs text-gray-600">"{tokens[i]}"</div>
                  <div className="text-sm font-bold text-purple-700">{klReward.toFixed(3)}</div>
                  <div className="text-xs text-gray-500">-β×KL[{i}]</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Final Reward */}
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <h4 className="font-semibold mb-3 text-green-700">Final Reward</h4>
          <div className="text-center">
            <div className="bg-white border-2 border-green-400 rounded-lg p-4 inline-block">
              <div className="text-sm text-gray-600">End of sequence</div>
              <div className="text-2xl font-bold text-green-700">{reward.toFixed(1)}</div>
              <div className="text-xs text-gray-500">Human feedback reward</div>
            </div>
          </div>
        </div>

        {/* Combined Scores */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-300">
          <h4 className="font-semibold mb-3 text-gray-700">Combined Score Vector</h4>
          <div className="grid grid-cols-5 gap-2 max-w-4xl mx-auto">
            {scores.map((score, i) => (
              <div key={i} className="text-center">
                <div className={`border-2 rounded-lg p-2 ${
                  i === scores.length - 1 
                    ? 'bg-green-100 border-green-400' 
                    : 'bg-purple-100 border-purple-400'
                }`}>
                  <div className="text-xs text-gray-600">
                    {i === scores.length - 1 ? 'Reward' : `"${tokens[i]}"`}
                  </div>
                  <div className="text-sm font-bold">{score.toFixed(3)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-300 rounded">
        <div className="text-sm text-yellow-800">
          <strong>Formula:</strong> score = cat([-β × KL, reward]) where β = {klBeta}
        </div>
      </div>
    </div>
  );

  // Values Visualization
  const ValuesVisualization = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-green-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-green-800">Value Model Predictions</h3>
      
      <div className="grid grid-cols-5 gap-2 mb-4 max-w-4xl mx-auto">
        {values.map((value, i) => (
          <div key={i} className="text-center">
            <div className="bg-green-100 border-2 border-green-400 rounded-lg p-3">
              <div className="text-xs text-gray-600">
                {i < completion.length ? `"${completion[i]}"` : 'End'}
              </div>
              <div className="text-lg font-bold text-green-700">{value.toFixed(1)}</div>
              <div className="text-xs text-gray-500">V({i})</div>
            </div>
          </div>
        ))}
      </div>

      <div className="p-3 bg-green-50 border border-green-300 rounded">
        <div className="text-sm text-green-800">
          <strong>Note:</strong> Values predict expected future rewards from each position
        </div>
      </div>
    </div>
  );

  // GAE Calculation Visualization
  const GAEVisualization = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-orange-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-orange-800">GAE Advantage Calculation</h3>
      
      <div className="mb-6">
        <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
          <h4 className="font-semibold mb-2 text-orange-700">Parameters</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>γ (gamma) = {gamma}</div>
            <div>λ (lambda) = {lambda}</div>
          </div>
        </div>
      </div>

      {/* Complete sequence display */}
      <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h4 className="font-semibold mb-2 text-blue-700">Complete Sequence</h4>
        <div className="flex flex-wrap gap-1 text-sm">
          {completion.map((token, i) => (
            <span key={i} className="bg-white border border-blue-300 rounded px-2 py-1">
              {token}
            </span>
          ))}
        </div>
      </div>

      {/* Step-by-step calculation */}
      <div className="space-y-3">
        <h4 className="font-semibold text-orange-700">Backward Calculation (t = {scores.length - 1} to 0)</h4>
        
        {scores.map((score, t) => {
          const reverseT = scores.length - 1 - t;
          const isActive = animatedToken === reverseT;
          const nextValue = reverseT < scores.length - 1 ? values[reverseT + 1] : 0.0;
          const delta = scores[reverseT] + gamma * nextValue - values[reverseT];
          
          return (
            <div key={reverseT} className={`p-3 rounded-lg border-2 transition-all duration-500 ${
              isActive ? 'border-orange-500 bg-orange-100 scale-105' : 'border-gray-300 bg-gray-50'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <div className="font-medium text-sm">
                  Step t = {reverseT} {reverseT < completion.length ? `("${completion[reverseT]}")` : '(end)'}
                </div>
                {isActive && <Calculator className="w-4 h-4 text-orange-600 animate-bounce" />}
              </div>
              
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="space-y-1">
                  <div className="bg-white p-2 rounded border">
                    <div className="text-gray-600">Score[{reverseT}]:</div>
                    <div className="font-bold">{scores[reverseT].toFixed(3)}</div>
                  </div>
                  <div className="bg-white p-2 rounded border">
                    <div className="text-gray-600">Value[{reverseT}]:</div>
                    <div className="font-bold">{values[reverseT].toFixed(3)}</div>
                  </div>
                  <div className="bg-white p-2 rounded border">
                    <div className="text-gray-600">Next Value:</div>
                    <div className="font-bold">{nextValue.toFixed(3)}</div>
                  </div>
                </div>
                
                <div className="space-y-1">
                  <div className="bg-blue-100 p-2 rounded border border-blue-300">
                    <div className="text-blue-700 text-xs">δ = r + γV_{t+1} - V_t</div>
                    <div className="font-bold text-blue-800">{delta.toFixed(3)}</div>
                  </div>
                  <div className="bg-yellow-100 p-2 rounded border border-yellow-300">
                    <div className="text-yellow-700 text-xs">A_t = δ + γλA_{t+1}</div>
                    <div className="font-bold text-yellow-800">{advantages[reverseT].toFixed(3)}</div>
                  </div>
                  <div className="bg-green-100 p-2 rounded border border-green-300">
                    <div className="text-green-700 text-xs">Return = A_t + V_t</div>
                    <div className="font-bold text-green-800">{returns[reverseT].toFixed(3)}</div>
                  </div>
                </div>
              </div>
            </div>
          );
        }).reverse()}
      </div>

      <div className="mt-6 p-4 bg-orange-50 border border-orange-300 rounded">
        <h4 className="font-semibold mb-2 text-orange-800">Final Results</h4>
        <div className="space-y-3">
          <div>
            <div className="text-sm font-medium text-orange-700 mb-2">Advantages (A_t):</div>
            <div className="grid grid-cols-5 gap-1">
              {advantages.map((adv, i) => (
                <div key={i} className="bg-white border border-orange-300 rounded p-2 text-center">
                  <div className="text-xs text-gray-600">"{completion[i] || 'end'}"</div>
                  <div className="font-bold text-xs">{adv.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-sm font-medium text-orange-700 mb-2">Returns (G_t):</div>
            <div className="grid grid-cols-5 gap-1">
              {returns.map((ret, i) => (
                <div key={i} className="bg-white border border-orange-300 rounded p-2 text-center">
                  <div className="text-xs text-gray-600">"{completion[i] || 'end'}"</div>
                  <div className="font-bold text-xs">{ret.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center mb-2 text-gray-800">
          Advantage Calculation in RLHF
        </h1>
        <p className="text-center text-gray-600 mb-4">
          Understanding GAE (Generalized Advantage Estimation) for the complete sequence
        </p>
        <p className="text-center text-lg font-mono bg-gray-100 p-2 rounded mb-8">
          "Where is Pune? Pune is in India."
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
          <div className="text-green-400 mb-2"># Advantage calculation process:</div>
          <div className={`${currentStep >= 0 ? 'text-yellow-400' : 'text-gray-500'}`}>
            original_log_prob = model.log_probs(completion_minus_1, target)
          </div>
          <div className={`${currentStep >= 0 ? 'text-yellow-400' : 'text-gray-500'}`}>
            ref_log_prob = ref_model.log_probs(completion_minus_1, target)
          </div>
          <div className={`${currentStep >= 1 ? 'text-yellow-400' : 'text-gray-500'}`}>
            kl = original_log_prob - ref_log_prob
          </div>
          <div className={`${currentStep >= 1 ? 'text-yellow-400' : 'text-gray-500'}`}>
            score = torch.cat((-kl_beta * kl, reward), dim=1)
          </div>
          <div className={`${currentStep >= 2 ? 'text-yellow-400' : 'text-gray-500'}`}>
            values = value_model(completion)
          </div>
          <div className={`${currentStep >= 3 ? 'text-yellow-400' : 'text-gray-500'}`}>
            advantage, returns = calculate_advantage_and_returns(score, values, gamma, lambd)
          </div>
        </div>

        {/* Visualizations */}
        <div className="space-y-8">
          {currentStep >= 0 && (
            <div className={`transition-all duration-500 ${currentStep === 0 ? 'ring-2 ring-blue-500' : ''}`}>
              <LogProbsVisualization />
            </div>
          )}

          {currentStep >= 1 && (
            <div className={`transition-all duration-500 ${currentStep === 1 ? 'ring-2 ring-blue-500' : ''}`}>
              <ScoresVisualization />
            </div>
          )}

          {currentStep >= 2 && (
            <div className={`transition-all duration-500 ${currentStep === 2 ? 'ring-2 ring-blue-500' : ''}`}>
              <ValuesVisualization />
            </div>
          )}

          {currentStep >= 3 && (
            <div className={`transition-all duration-500 ${currentStep === 3 ? 'ring-2 ring-blue-500' : ''}`}>
              <GAEVisualization />
            </div>
          )}
        </div>

        {/* Summary */}
        {currentStep >= 3 && (
          <div className="mt-8 p-6 bg-indigo-50 rounded-lg border border-indigo-200">
            <h3 className="text-lg font-semibold text-indigo-800 mb-3">Key Insights:</h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-indigo-700">
              <div>
                <strong>Complete Sequence Processing:</strong> All 9 tokens are now included in the advantage calculation
              </div>
              <div>
                <strong>GAE Calculation:</strong> Estimates advantages by bootstrapping from future values for every position
              </div>
              <div>
                <strong>Backward Iteration:</strong> Starts from the end and works backwards to accumulate advantages
              </div>
              <div>
                <strong>Token-Level Granularity:</strong> Each token gets its own advantage estimate for precise policy updates
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdvantageCalculationViz;