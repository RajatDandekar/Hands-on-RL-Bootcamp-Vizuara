import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, Square, Circle, ArrowRight } from 'lucide-react';

const PaddingMaskingViz = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [animatedIndex, setAnimatedIndex] = useState(-1);

  // Example data - "Where is Pune? Pune is in India."
  const prompt = ["Where", " is", " Pune", "?"];
  const completion = ["Where", " is", " Pune", "?", " Pune", " is", " in", " India", "."];
  const target = [" is", " Pune", "?", " Pune", " is", " in", " India", "."];
  
  const block_size = 16; // Maximum sequence length for the batch
  
  // Simulated data arrays (before padding)
  const original_advantage = [0.12, 0.34, -0.15, 0.67, 0.89, 0.23, -0.45, 0.78, 0.91];
  const original_returns = [1.2, 1.4, 1.1, 1.8, 2.1, 2.3, 2.0, 2.7, 2.9];
  const original_log_probs = [-1.2, -0.8, -2.1, -1.5, -0.9, -0.7, -1.8, -1.1]; // 8 values (completion.length - 1)

  const steps = [
    {
      title: "1. Original Data",
      description: "Start with unpadded sequences of different lengths"
    },
    {
      title: "2. Pad Advantages & Returns",
      description: "Pad advantages and returns to block_size with zeros"
    },
    {
      title: "3. Pad Log Probabilities", 
      description: "Pad log probabilities to block_size with zeros"
    },
    {
      title: "4. Pad Token Sequences",
      description: "Pad completion and target sequences with zeros"
    },
    {
      title: "5. Create Action Mask",
      description: "Create mask to identify generated tokens vs padding"
    }
  ];

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      setAnimatedIndex(-1);
    }
  };

  const reset = () => {
    setCurrentStep(0);
    setAnimatedIndex(-1);
  };

  // Auto-animate padding process
  useEffect(() => {
    if (currentStep > 0) {
      const interval = setInterval(() => {
        setAnimatedIndex(prev => (prev + 1) % (block_size + 2));
      }, 300);
      return () => clearInterval(interval);
    }
  }, [currentStep, block_size]);

  // Helper function to create padded arrays
  const createPaddedArray = (originalArray, targetLength, fillValue = 0) => {
    const padded = [...originalArray];
    while (padded.length < targetLength) {
      padded.push(fillValue);
    }
    return padded;
  };

  // Padded data
  const padded_advantages = createPaddedArray(original_advantage, block_size, 0);
  const padded_returns = createPaddedArray(original_returns, block_size, 0);
  const padded_log_probs = createPaddedArray(original_log_probs, block_size, 0);
  const padded_completion = createPaddedArray(completion, block_size, "<PAD>");
  const padded_target = createPaddedArray(target, block_size, "<PAD>");
  
  // Action mask: 1 for generated tokens, 0 for prompt and padding
  const action_mask = Array(block_size).fill(0);
  for (let i = prompt.length; i < completion.length; i++) {
    action_mask[i] = 1;
  }

  // Visualization Components
  const TokenBox = ({ token, index, isHighlighted, isPadding, isPrompt, isGenerated, showMask = false, maskValue = 0 }) => {
    const getBoxStyle = () => {
      if (isPadding) return "bg-gray-200 border-gray-400 text-gray-500";
      if (isPrompt) return "bg-blue-100 border-blue-400 text-blue-800";
      if (isGenerated) return "bg-green-100 border-green-400 text-green-800";
      return "bg-white border-gray-300 text-gray-700";
    };

    const getMaskStyle = () => {
      return maskValue === 1 ? "bg-yellow-300 border-yellow-500" : "bg-gray-300 border-gray-500";
    };

    return (
      <div className={`relative transition-all duration-300 ${
        isHighlighted ? 'scale-110 ring-2 ring-orange-400' : ''
      }`}>
        <div className={`border-2 rounded-lg p-2 text-center min-h-[60px] flex flex-col justify-center ${getBoxStyle()}`}>
          <div className="text-xs font-mono truncate">
            {typeof token === 'string' ? `"${token}"` : token.toFixed(2)}
          </div>
          <div className="text-xs text-gray-500">{index}</div>
        </div>
        {showMask && (
          <div className={`mt-1 border-2 rounded p-1 text-center text-xs ${getMaskStyle()}`}>
            {maskValue}
          </div>
        )}
      </div>
    );
  };

  const ArrayVisualization = ({ title, data, originalLength, showMask = false, maskData = [] }) => (
    <div className="bg-white p-4 rounded-lg border-2 border-gray-300 shadow-lg mb-4">
      <h4 className="font-semibold mb-3 text-gray-800">{title}</h4>
      <div className="grid grid-cols-8 gap-2">
        {data.slice(0, block_size).map((item, index) => (
          <TokenBox
            key={index}
            token={item}
            index={index}
            isHighlighted={animatedIndex === index}
            isPadding={index >= originalLength}
            isPrompt={index < prompt.length}
            isGenerated={index >= prompt.length && index < completion.length}
            showMask={showMask}
            maskValue={maskData[index] || 0}
          />
        ))}
      </div>
      <div className="mt-2 text-xs text-gray-600">
        Original length: {originalLength}, Padded to: {block_size}
      </div>
    </div>
  );

  const OriginalDataView = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 p-6 rounded-lg border-2 border-blue-300">
        <h3 className="text-lg font-bold mb-4 text-blue-800">Original Unpadded Data</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Prompt (length: {prompt.length})</h4>
              <div className="flex space-x-1">
                {prompt.map((token, i) => (
                  <div key={i} className="bg-blue-100 border border-blue-300 rounded px-2 py-1 text-sm">
                    "{token}"
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Completion (length: {completion.length})</h4>
              <div className="flex flex-wrap gap-1">
                {completion.map((token, i) => (
                  <div key={i} className={`border rounded px-2 py-1 text-sm ${
                    i < prompt.length ? 'bg-blue-100 border-blue-300' : 'bg-green-100 border-green-300'
                  }`}>
                    "{token}"
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Target (length: {target.length})</h4>
              <div className="flex flex-wrap gap-1">
                {target.map((token, i) => (
                  <div key={i} className="bg-green-100 border border-green-300 rounded px-2 py-1 text-sm">
                    "{token}"
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Advantages (length: {original_advantage.length})</h4>
              <div className="grid grid-cols-3 gap-1 text-xs">
                {original_advantage.map((val, i) => (
                  <div key={i} className="bg-orange-100 border border-orange-300 rounded p-1 text-center">
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Returns (length: {original_returns.length})</h4>
              <div className="grid grid-cols-3 gap-1 text-xs">
                {original_returns.map((val, i) => (
                  <div key={i} className="bg-purple-100 border border-purple-300 rounded p-1 text-center">
                    {val.toFixed(1)}
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-semibold mb-2">Log Probs (length: {original_log_probs.length})</h4>
              <div className="grid grid-cols-4 gap-1 text-xs">
                {original_log_probs.map((val, i) => (
                  <div key={i} className="bg-red-100 border border-red-300 rounded p-1 text-center">
                    {val.toFixed(1)}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-300 rounded">
          <div className="text-sm text-yellow-800">
            <strong>Problem:</strong> Different sequences have different lengths. 
            Batching requires all sequences to have the same length (block_size = {block_size}).
          </div>
        </div>
      </div>
    </div>
  );

  const PaddedDataView = () => (
    <div className="space-y-6">
      {currentStep >= 1 && (
        <ArrayVisualization
          title="Padded Advantages"
          data={padded_advantages}
          originalLength={original_advantage.length}
        />
      )}

      {currentStep >= 1 && (
        <ArrayVisualization
          title="Padded Returns"
          data={padded_returns}
          originalLength={original_returns.length}
        />
      )}

      {currentStep >= 2 && (
        <ArrayVisualization
          title="Padded Log Probabilities"
          data={padded_log_probs}
          originalLength={original_log_probs.length}
        />
      )}

      {currentStep >= 3 && (
        <ArrayVisualization
          title="Padded Completion Tokens"
          data={padded_completion}
          originalLength={completion.length}
        />
      )}

      {currentStep >= 3 && (
        <ArrayVisualization
          title="Padded Target Tokens"
          data={padded_target}
          originalLength={target.length}
        />
      )}

      {currentStep >= 4 && (
        <ArrayVisualization
          title="Action Mask"
          data={action_mask}
          originalLength={block_size}
          showMask={true}
          maskData={action_mask}
        />
      )}
    </div>
  );

  const CodeVisualization = () => (
    <div className="bg-gray-900 text-white p-4 rounded-lg font-mono text-sm mb-6">
      <div className="text-green-400 mb-2"># Padding and Masking Process:</div>
      
      <div className={`${currentStep >= 1 ? 'text-yellow-400' : 'text-gray-500'}`}>
        <div># Step 1: Pad advantages and returns</div>
        <div>pad = torch.zeros(1, block_size - advantage.size(1))</div>
        <div>advantages.append(torch.cat((advantage, pad), dim=1))</div>
        <div>returns.append(torch.cat((single_return, pad), dim=1))</div>
      </div>
      
      <div className={`mt-2 ${currentStep >= 2 ? 'text-yellow-400' : 'text-gray-500'}`}>
        <div># Step 2: Pad log probabilities</div>
        <div>pad_plus_1 = torch.zeros(1, block_size - original_log_prob.size(1))</div>
        <div>original_log_probs.append(torch.cat((original_log_prob, pad_plus_1), dim=1))</div>
      </div>
      
      <div className={`mt-2 ${currentStep >= 3 ? 'text-yellow-400' : 'text-gray-500'}`}>
        <div># Step 3: Pad token sequences</div>
        <div>pad = torch.zeros(1, block_size - completion.size(1))</div>
        <div>completions.append(torch.cat((completion, pad), dim=1))</div>
        <div>targets.append(torch.cat((target, pad), dim=1))</div>
      </div>
      
      <div className={`mt-2 ${currentStep >= 4 ? 'text-yellow-400' : 'text-gray-500'}`}>
        <div># Step 4: Create action mask</div>
        <div>mask = torch.zeros(1, block_size)</div>
        <div>mask[:, prompt.size(1):completion.size(1)] = 1</div>
        <div>action_mask.append(mask)</div>
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center mb-2 text-gray-800">
          RLHF Padding and Masking Process
        </h1>
        <p className="text-center text-gray-600 mb-4">
          Understanding how sequences are padded and masked for batch processing
        </p>
        <p className="text-center text-lg font-mono bg-gray-100 p-2 rounded mb-8">
          Example: "Where is Pune? Pune is in India." → block_size = {block_size}
        </p>

        {/* Controls */}
        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={nextStep}
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

        <CodeVisualization />

        {/* Legend */}
        <div className="mb-6 p-4 bg-gray-50 rounded-lg border">
          <h3 className="font-semibold mb-3">Legend:</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-100 border border-blue-300 rounded"></div>
              <span>Prompt tokens</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-green-100 border border-green-300 rounded"></div>
              <span>Generated tokens</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-gray-200 border border-gray-400 rounded"></div>
              <span>Padding</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-yellow-300 border border-yellow-500 rounded"></div>
              <span>Action mask = 1</span>
            </div>
          </div>
        </div>

        {/* Main visualization */}
        {currentStep === 0 ? <OriginalDataView /> : <PaddedDataView />}

        {/* Summary */}
        {currentStep >= 4 && (
          <div className="mt-8 p-6 bg-indigo-50 rounded-lg border border-indigo-200">
            <h3 className="text-lg font-semibold text-indigo-800 mb-3">Key Insights:</h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-indigo-700">
              <div>
                <strong>Padding Purpose:</strong> Ensures all sequences in a batch have the same length for efficient GPU processing
              </div>
              <div>
                <strong>Action Mask:</strong> Identifies which positions correspond to generated tokens vs prompt/padding
              </div>
              <div>
                <strong>Loss Calculation:</strong> Only positions with mask=1 contribute to the policy gradient loss
              </div>
              <div>
                <strong>Memory Efficiency:</strong> Padding allows vectorized operations across the entire batch
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-indigo-100 rounded border border-indigo-300">
              <div className="text-sm text-indigo-800">
                <strong>Action Mask Pattern:</strong> [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0]<br/>
                Positions 0-3: Prompt (mask=0), Positions 4-8: Generated (mask=1), Positions 9-15: Padding (mask=0)
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PaddingMaskingViz;