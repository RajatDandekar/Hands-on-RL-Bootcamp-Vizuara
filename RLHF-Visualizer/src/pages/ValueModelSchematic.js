import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, ArrowDown, ArrowRight, Zap, Brain, Hash, TrendingUp, Circle } from 'lucide-react';

const ValueModelSchematic = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [animationStep, setAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // Example token journey
  const exampleToken = "Where";
  const tokenId = 2940;
  
  // Simulated intermediate representations
  const embedding = [0.1, -0.3, 0.8, 0.2, -0.1, 0.7, -0.4, 0.5];
  const transformerOutput = [0.2, 0.6, -0.2, 0.9, 0.1, -0.3, 0.4, -0.1];
  const finalValue = 0.73;

  const steps = [
    {
      title: "Token Input & Embedding",
      description: "Follow the journey of token 'Where' as it gets converted to embeddings"
    },
    {
      title: "Transformer Processing", 
      description: "Watch how the token flows through attention and feed-forward layers"
    },
    {
      title: "Value Prediction Network",
      description: "See the detailed neural network that converts 768 dimensions to 1 value"
    },
    {
      title: "Final Value Output",
      description: "The complete journey results in a single scalar value"
    }
  ];

  const animateStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      setAnimationStep(0);
    }
  };

  const reset = () => {
    setCurrentStep(0);
    setAnimationStep(0);
    setIsAnimating(false);
  };

  // Auto-animate within each step
  useEffect(() => {
    if (currentStep > 0) {
      const interval = setInterval(() => {
        setAnimationStep(prev => (prev + 1) % 4);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [currentStep]);

  // Token flow animation component
  const FlowingToken = ({ from, to, active, delay = 0 }) => {
    if (!active) return null;
    
    return (
      <div 
        className="absolute w-4 h-4 bg-blue-500 rounded-full animate-pulse"
        style={{
          animation: `tokenFlow 2s ease-in-out infinite`,
          animationDelay: `${delay}s`,
          left: from.x,
          top: from.y,
          transform: 'translate(-50%, -50%)'
        }}
      >
        <style jsx>{`
          @keyframes tokenFlow {
            0% { left: ${from.x}px; top: ${from.y}px; }
            50% { left: ${(from.x + to.x) / 2}px; top: ${(from.y + to.y) / 2}px; }
            100% { left: ${to.x}px; top: ${to.y}px; }
          }
        `}</style>
      </div>
    );
  };

  // Detailed embedding visualization
  const EmbeddingVisualization = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-blue-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-blue-800">Token Embedding Process</h3>
      
      <div className="flex items-center justify-center space-x-8">
        {/* Token Input */}
        <div className="text-center">
          <div className="w-20 h-20 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-lg mb-2">
            {tokenId}
          </div>
          <div className="font-medium">"{exampleToken}"</div>
          <div className="text-sm text-gray-500">Token ID</div>
        </div>

        <ArrowRight className="w-8 h-8 text-gray-400" />

        {/* Embedding Table */}
        <div className="text-center">
          <div className="bg-gray-100 p-4 rounded border">
            <div className="text-sm font-medium mb-2">Embedding Table</div>
            <div className="grid grid-cols-4 gap-1">
              {[...Array(16)].map((_, i) => (
                <div key={i} className="w-3 h-3 bg-blue-300 rounded"></div>
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-1">[50k × 768]</div>
          </div>
        </div>

        <ArrowRight className="w-8 h-8 text-gray-400" />

        {/* Output Vector */}
        <div className="text-center">
          <div className="bg-green-100 p-3 rounded border">
            <div className="text-sm font-medium mb-2">Dense Vector</div>
            <div className="flex space-x-1">
              {embedding.map((val, i) => (
                <div key={i} className={`w-8 h-12 rounded flex items-end justify-center text-xs ${
                  val > 0 ? 'bg-green-400' : 'bg-red-400'
                }`}>
                  <div className="text-white font-bold" style={{
                    height: `${Math.abs(val) * 40 + 10}px`,
                    display: 'flex',
                    alignItems: 'center'
                  }}>
                    {val.toFixed(1)}
                  </div>
                </div>
              ))}
              <div className="text-xs">...</div>
            </div>
            <div className="text-xs text-gray-500 mt-1">[768 dims]</div>
          </div>
        </div>
      </div>
    </div>
  );

  // Detailed transformer visualization
  const TransformerVisualization = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-purple-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-purple-800">Transformer Layer Processing</h3>
      
      <div className="space-y-6">
        {/* Attention Mechanism */}
        <div className="border-2 border-purple-200 rounded-lg p-4">
          <h4 className="font-semibold mb-3 text-purple-700">Multi-Head Attention</h4>
          <div className="flex justify-center items-center space-x-4">
            {/* Query, Key, Value */}
            <div className="text-center">
              <div className="space-y-2">
                {['Q', 'K', 'V'].map((label, i) => (
                  <div key={label} className="w-12 h-8 bg-purple-200 rounded flex items-center justify-center font-bold">
                    {label}
                  </div>
                ))}
              </div>
              <div className="text-xs text-gray-500 mt-2">Linear Projections</div>
            </div>

            <ArrowRight className="w-6 h-6 text-gray-400" />

            {/* Attention Matrix */}
            <div className="text-center">
              <div className="grid grid-cols-4 gap-1 mb-2">
                {[...Array(16)].map((_, i) => (
                  <div key={i} className={`w-4 h-4 rounded ${
                    i % 5 === 0 ? 'bg-yellow-400' : 'bg-purple-100'
                  }`}></div>
                ))}
              </div>
              <div className="text-xs text-gray-500">Attention Weights</div>
            </div>

            <ArrowRight className="w-6 h-6 text-gray-400" />

            {/* Output */}
            <div className="text-center">
              <div className="w-16 h-12 bg-purple-300 rounded flex items-center justify-center">
                <Brain className="w-8 h-8 text-purple-700" />
              </div>
              <div className="text-xs text-gray-500 mt-2">Attended Output</div>
            </div>
          </div>
        </div>

        {/* Feed Forward Network */}
        <div className="border-2 border-purple-200 rounded-lg p-4">
          <h4 className="font-semibold mb-3 text-purple-700">Feed Forward Network</h4>
          <div className="flex justify-center items-center space-x-6">
            {/* Input Layer */}
            <div className="flex flex-col space-y-1">
              <div className="text-xs text-center mb-1">768</div>
              {[...Array(8)].map((_, i) => (
                <div key={i} className="w-3 h-3 rounded-full bg-purple-400"></div>
              ))}
              <div className="text-xs text-center">...</div>
            </div>

            {/* Hidden Layer */}
            <div className="flex flex-col space-y-1">
              <div className="text-xs text-center mb-1">3072</div>
              {[...Array(12)].map((_, i) => (
                <div key={i} className="w-3 h-3 rounded-full bg-yellow-400"></div>
              ))}
              <div className="text-xs text-center">...</div>
            </div>

            {/* Output Layer */}
            <div className="flex flex-col space-y-1">
              <div className="text-xs text-center mb-1">768</div>
              {[...Array(8)].map((_, i) => (
                <div key={i} className="w-3 h-3 rounded-full bg-purple-400"></div>
              ))}
              <div className="text-xs text-center">...</div>
            </div>
          </div>
          <div className="text-center mt-2 text-xs text-gray-500">
            Linear → ReLU → Linear
          </div>
        </div>
      </div>
    </div>
  );

  // Detailed value prediction network
  const ValuePredictionNetwork = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-green-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-green-800">Value Prediction Network</h3>
      
      <div className="flex items-center justify-center space-x-8">
        {/* Input Layer (768 neurons) */}
        <div className="text-center">
          <div className="text-sm font-medium mb-3">Transformer Output</div>
          <div className="flex flex-col space-y-1">
            {transformerOutput.map((val, i) => (
              <div key={i} className="flex items-center space-x-2">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                  val > 0 ? 'bg-green-400 text-white' : 'bg-red-400 text-white'
                }`}>
                  {val.toFixed(1)}
                </div>
                {animationStep % 2 === 0 && (
                  <div className="w-8 h-px bg-blue-500 animate-pulse"></div>
                )}
              </div>
            ))}
            <div className="text-xs text-gray-500">... (768 total)</div>
          </div>
          <div className="text-xs text-gray-500 mt-2">[768 dimensions]</div>
        </div>

        {/* Weight Matrix Visualization */}
        <div className="text-center">
          <div className="text-sm font-medium mb-3">Weight Matrix</div>
          <div className="bg-gray-100 p-3 rounded border">
            <div className="grid grid-cols-8 gap-px">
              {[...Array(64)].map((_, i) => (
                <div key={i} className={`w-2 h-2 ${
                  Math.random() > 0.5 ? 'bg-blue-400' : 'bg-red-400'
                } ${animationStep === 1 ? 'animate-pulse' : ''}`}></div>
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-2">W: [768 × 1]</div>
          </div>
          
          {/* Mathematical operation */}
          <div className="mt-3 p-2 bg-yellow-100 rounded border text-xs">
            <div className="font-mono">y = W·x + b</div>
            <div className="text-gray-600">Matrix multiplication</div>
          </div>
        </div>

        {/* Output Neuron */}
        <div className="text-center">
          <div className="text-sm font-medium mb-3">Value Output</div>
          <div className={`w-16 h-16 rounded-full bg-green-500 flex items-center justify-center text-white font-bold text-lg ${
            animationStep === 2 ? 'animate-bounce' : ''
          }`}>
            {finalValue}
          </div>
          <div className="text-xs text-gray-500 mt-2">Scalar Value</div>
          
          {/* Value interpretation */}
          <div className="mt-3 p-2 bg-green-100 rounded border text-xs">
            <div className="font-medium">Expected Reward</div>
            <div className="text-gray-600">From this position</div>
          </div>
        </div>
      </div>

      {/* Detailed computation steps */}
      <div className="mt-6 p-4 bg-gray-50 rounded border">
        <h4 className="font-semibold mb-2">Computation Steps:</h4>
        <div className="space-y-2 text-sm font-mono">
          <div className={`${animationStep >= 0 ? 'text-blue-600 font-bold' : 'text-gray-500'}`}>
            1. input = transformer_output  # [768]
          </div>
          <div className={`${animationStep >= 1 ? 'text-blue-600 font-bold' : 'text-gray-500'}`}>
            2. weighted_sum = W @ input    # [1]  
          </div>
          <div className={`${animationStep >= 2 ? 'text-blue-600 font-bold' : 'text-gray-500'}`}>
            3. value = weighted_sum + bias # [1]
          </div>
          <div className={`${animationStep >= 3 ? 'text-blue-600 font-bold' : 'text-gray-500'}`}>
            4. return value.squeeze()     # scalar
          </div>
        </div>
      </div>
    </div>
  );

  // Complete token journey visualization
  const TokenJourneyVisualization = () => (
    <div className="bg-white p-6 rounded-lg border-2 border-indigo-300 shadow-lg">
      <h3 className="text-lg font-bold text-center mb-4 text-indigo-800">Complete Token Journey</h3>
      
      <div className="relative">
        {/* Journey path */}
        <div className="flex items-center justify-between space-x-4">
          {/* Stage 1: Token */}
          <div className="text-center">
            <div className={`w-16 h-16 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold ${
              animationStep === 0 ? 'ring-4 ring-blue-300 animate-pulse' : ''
            }`}>
              <Hash className="w-8 h-8" />
            </div>
            <div className="text-sm font-medium mt-2">Token</div>
            <div className="text-xs text-gray-500">"{exampleToken}"</div>
          </div>

          <ArrowRight className="w-6 h-6 text-gray-400" />

          {/* Stage 2: Embedding */}
          <div className="text-center">
            <div className={`w-16 h-16 rounded-full bg-green-500 flex items-center justify-center text-white font-bold ${
              animationStep === 1 ? 'ring-4 ring-green-300 animate-pulse' : ''
            }`}>
              <Circle className="w-8 h-8" />
            </div>
            <div className="text-sm font-medium mt-2">Embedding</div>
            <div className="text-xs text-gray-500">[768] vector</div>
          </div>

          <ArrowRight className="w-6 h-6 text-gray-400" />

          {/* Stage 3: Transformer */}
          <div className="text-center">
            <div className={`w-16 h-16 rounded-full bg-purple-500 flex items-center justify-center text-white font-bold ${
              animationStep === 2 ? 'ring-4 ring-purple-300 animate-pulse' : ''
            }`}>
              <Brain className="w-8 h-8" />
            </div>
            <div className="text-sm font-medium mt-2">Transformer</div>
            <div className="text-xs text-gray-500">12 layers</div>
          </div>

          <ArrowRight className="w-6 h-6 text-gray-400" />

          {/* Stage 4: Value Head */}
          <div className="text-center">
            <div className={`w-16 h-16 rounded-full bg-yellow-500 flex items-center justify-center text-white font-bold ${
              animationStep === 3 ? 'ring-4 ring-yellow-300 animate-pulse' : ''
            }`}>
              <Zap className="w-8 h-8" />
            </div>
            <div className="text-sm font-medium mt-2">Value Head</div>
            <div className="text-xs text-gray-500">Linear [768→1]</div>
          </div>

          <ArrowRight className="w-6 h-6 text-gray-400" />

          {/* Stage 5: Final Value */}
          <div className="text-center">
            <div className={`w-16 h-16 rounded-full bg-red-500 flex items-center justify-center text-white font-bold ${
              animationStep === 0 ? 'ring-4 ring-red-300 animate-pulse' : ''
            }`}>
              <TrendingUp className="w-8 h-8" />
            </div>
            <div className="text-sm font-medium mt-2">Value</div>
            <div className="text-xs text-gray-500">{finalValue}</div>
          </div>
        </div>

        {/* Data flow indicators */}
        <div className="mt-6 grid grid-cols-4 gap-4 text-xs">
          <div className="text-center p-2 bg-blue-50 rounded">
            <div className="font-medium">Input</div>
            <div>Token ID: {tokenId}</div>
          </div>
          <div className="text-center p-2 bg-green-50 rounded">
            <div className="font-medium">Embedded</div>
            <div>768-dim vector</div>
          </div>
          <div className="text-center p-2 bg-purple-50 rounded">
            <div className="font-medium">Processed</div>
            <div>Contextualized</div>
          </div>
          <div className="text-center p-2 bg-red-50 rounded">
            <div className="font-medium">Output</div>
            <div>Scalar: {finalValue}</div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center mb-2 text-gray-800">
          Value Model: Token Journey Visualization
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Follow the complete journey of a token through the Value Model architecture
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

        {/* Journey Overview (always visible) */}
        <div className="mb-8">
          <TokenJourneyVisualization />
        </div>

        {/* Detailed visualizations based on current step */}
        <div className="space-y-8">
          {currentStep >= 0 && (
            <div className={`transition-all duration-500 ${currentStep === 0 ? 'ring-2 ring-blue-500' : ''}`}>
              <EmbeddingVisualization />
            </div>
          )}

          {currentStep >= 1 && (
            <div className={`transition-all duration-500 ${currentStep === 1 ? 'ring-2 ring-blue-500' : ''}`}>
              <TransformerVisualization />
            </div>
          )}

          {currentStep >= 2 && (
            <div className={`transition-all duration-500 ${currentStep === 2 ? 'ring-2 ring-blue-500' : ''}`}>
              <ValuePredictionNetwork />
            </div>
          )}

          {currentStep >= 3 && (
            <div className="p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-300">
              <h3 className="text-lg font-bold mb-4 text-green-800">Journey Complete!</h3>
              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div className="p-4 bg-white rounded border">
                  <div className="font-semibold text-blue-700">Input Token</div>
                  <div>"{exampleToken}" (ID: {tokenId})</div>
                </div>
                <div className="p-4 bg-white rounded border">
                  <div className="font-semibold text-purple-700">Processing</div>
                  <div>Embedding → Transformer → Linear</div>
                </div>
                <div className="p-4 bg-white rounded border">
                  <div className="font-semibold text-green-700">Final Value</div>
                  <div className="text-2xl font-bold">{finalValue}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Code correlation */}
        <div className="mt-8 p-6 bg-gray-900 text-white rounded-lg">
          <h3 className="text-lg font-bold mb-4 text-gray-200">Code Correlation:</h3>
          <div className="font-mono text-sm space-y-2">
            <div className={`${currentStep >= 0 ? 'text-green-400' : 'text-gray-500'}`}>
              # Step 1: Token embedding (handled by transformer)
            </div>
            <div className={`${currentStep >= 1 ? 'text-green-400' : 'text-gray-500'}`}>
              x = self.transformer(toks, attention_mask=attn_mask)  # [b, t, n_embd]
            </div>
            <div className={`${currentStep >= 2 ? 'text-green-400' : 'text-gray-500'}`}>
              # Step 2: Value prediction head
            </div>
            <div className={`${currentStep >= 2 ? 'text-green-400' : 'text-gray-500'}`}>
              rewards = self.prediction_head(x).squeeze(-1)  # [b, t]
            </div>
            <div className={`${currentStep >= 3 ? 'text-green-400' : 'text-gray-500'}`}>
              return rewards  # Final values for each token position
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ValueModelSchematic;