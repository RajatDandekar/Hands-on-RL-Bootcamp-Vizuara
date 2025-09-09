import React, { useState } from 'react';
import { ChevronRight, Play, RotateCcw } from 'lucide-react';

const TokenVisualization = () => {
    const [currentStep, setCurrentStep] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);
  
    // Example tokens (simplified tokenization)
    const prompt = "Where is Pune?";
    const response = " Pune is in India.";
    const fullSequence = prompt + response;
    
    // Simulate tokenization (in reality, tokens would be numbers)
    const tokens = ["Where", " is", " Pune", "?", " Pune", " is", " in", " India", "."];
    const tokenIds = [2940, 318, 24723, 30, 24723, 318, 287, 3794, 13]; // Example token IDs
    
    const steps = [
      {
        title: "1. Model Generation",
        description: "The model generates a completion including both prompt and response",
        highlight: "generation"
      },
      {
        title: "2. Token Sequence",
        description: "The complete sequence is tokenized into individual tokens",
        highlight: "tokens"
      },
      {
        title: "3. Input-Target Splitting",
        description: "Split into input sequence (all but last) and target sequence (all but first)",
        highlight: "split"
      },
      {
        title: "4. Training Pairs",
        description: "Each input token predicts the next target token for training",
        highlight: "pairs"
      }
    ];
  
    const getInputTokens = () => tokens.slice(0, -1);
    const getTargetTokens = () => tokens.slice(1);
  
    const animateStep = () => {
      if (currentStep < steps.length - 1) {
        setIsAnimating(true);
        setTimeout(() => {
          setCurrentStep(currentStep + 1);
          setIsAnimating(false);
        }, 300);
      }
    };
  
    const reset = () => {
      setCurrentStep(0);
      setIsAnimating(false);
    };
  
    return (
      <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h1 className="text-3xl font-bold text-center mb-2 text-gray-800">
            Language Model Token Shifting
          </h1>
          <p className="text-center text-gray-600 mb-8">
            Understanding <code className="bg-gray-100 px-2 py-1 rounded">completion[:, :-1]</code> and <code className="bg-gray-100 px-2 py-1 rounded">completion[:, 1:]</code>
          </p>
  
          {/* Controls */}
          <div className="flex justify-center space-x-4 mb-8">
            <button
              onClick={animateStep}
              disabled={currentStep >= steps.length - 1 || isAnimating}
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
            <div>completion = model.generate(prompt, max_new_tokens=completion_len, ...)</div>
            <div className={`${currentStep >= 2 ? 'text-yellow-400' : 'text-gray-500'}`}>
              completion_minus_1, target = completion[:, :-1], completion[:, 1:]
            </div>
          </div>
  
          {/* Visualization */}
          <div className="space-y-8">
            {/* Step 1: Generation */}
            {currentStep >= 0 && (
              <div className={`transition-all duration-500 ${currentStep === 0 ? 'ring-2 ring-blue-500' : ''} bg-gray-50 p-6 rounded-lg`}>
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Generated Completion:</h3>
                <div className="bg-white p-4 rounded border-2 border-dashed border-gray-300">
                  <span className="text-blue-600 font-medium">Prompt:</span> "{prompt}"
                  <br />
                  <span className="text-green-600 font-medium">Response:</span> "{response}"
                  <br />
                  <span className="text-purple-600 font-medium">Full sequence:</span> "{fullSequence}"
                </div>
              </div>
            )}
  
            {/* Step 2: Tokenization */}
            {currentStep >= 1 && (
              <div className={`transition-all duration-500 ${currentStep === 1 ? 'ring-2 ring-blue-500' : ''} bg-gray-50 p-6 rounded-lg`}>
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Tokenized Sequence:</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                  {tokens.map((token, idx) => (
                    <div key={idx} className="bg-white border-2 border-gray-300 rounded-lg p-3 text-center">
                      <div className="font-mono text-sm bg-gray-100 px-2 py-1 rounded mb-1">
                        {tokenIds[idx]}
                      </div>
                      <div className="font-medium">"{token}"</div>
                      <div className="text-xs text-gray-500">pos {idx}</div>
                    </div>
                  ))}
                </div>
                <p className="text-sm text-gray-600">
                  Shape: [batch_size, {tokens.length}] - Each number represents a token ID
                </p>
              </div>
            )}
  
            {/* Step 3: Splitting */}
            {currentStep >= 2 && (
              <div className={`transition-all duration-500 ${currentStep === 2 ? 'ring-2 ring-blue-500' : ''} bg-gray-50 p-6 rounded-lg`}>
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Input-Target Splitting:</h3>
                
                {/* Input sequence */}
                <div className="mb-6">
                  <h4 className="font-medium mb-2 text-blue-600">
                    Input (completion[:, :-1]) - "All but last token"
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {getInputTokens().map((token, idx) => (
                      <div key={idx} className="bg-blue-100 border-2 border-blue-300 rounded-lg p-3 text-center">
                        <div className="font-mono text-sm bg-blue-200 px-2 py-1 rounded mb-1">
                          {tokenIds[idx]}
                        </div>
                        <div className="font-medium">"{token}"</div>
                        <div className="text-xs text-blue-700">pos {idx}</div>
                      </div>
                    ))}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">Shape: [batch_size, {getInputTokens().length}]</p>
                </div>
  
                {/* Target sequence */}
                <div>
                  <h4 className="font-medium mb-2 text-green-600">
                    Target (completion[:, 1:]) - "All but first token"
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {getTargetTokens().map((token, idx) => (
                      <div key={idx} className="bg-green-100 border-2 border-green-300 rounded-lg p-3 text-center">
                        <div className="font-mono text-sm bg-green-200 px-2 py-1 rounded mb-1">
                          {tokenIds[idx + 1]}
                        </div>
                        <div className="font-medium">"{token}"</div>
                        <div className="text-xs text-green-700">pos {idx}</div>
                      </div>
                    ))}
                  </div>
                  <p className="text-sm text-gray-600 mt-2">Shape: [batch_size, {getTargetTokens().length}]</p>
                </div>
              </div>
            )}
  
            {/* Step 4: Training pairs */}
            {currentStep >= 3 && (
              <div className={`transition-all duration-500 ${currentStep === 3 ? 'ring-2 ring-blue-500' : ''} bg-gray-50 p-6 rounded-lg`}>
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Training Pairs:</h3>
                <p className="text-gray-600 mb-4">Each input token should predict the corresponding target token:</p>
                
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
  
                <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h4 className="font-semibold text-yellow-800 mb-2">Why this matters for training:</h4>
                  <ul className="text-sm text-yellow-700 space-y-1">
                    <li>• The model learns to predict the next token given previous context</li>
                    <li>• Loss is computed between predicted probabilities and actual target tokens</li>
                    <li>• This creates {getInputTokens().length} training examples from one sequence</li>
                    <li>• Each position learns from increasingly more context (autoregressive)</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
  
          {/* Summary */}
          {currentStep >= 3 && (
            <div className="mt-8 p-6 bg-indigo-50 rounded-lg border border-indigo-200">
              <h3 className="text-lg font-semibold text-indigo-800 mb-3">Key Insights:</h3>
              <div className="grid md:grid-cols-2 gap-4 text-sm text-indigo-700">
                <div>
                  <strong>completion[:, :-1]</strong> creates the input sequence by removing the last token
                </div>
                <div>
                  <strong>completion[:, 1:]</strong> creates the target sequence by removing the first token
                </div>
                <div>
                  This shifting creates aligned input-target pairs for next-token prediction
                </div>
                <div>
                  Every token (except the first) serves as both a prediction target and future context
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
};

export default TokenVisualization;