import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import {
  Home,
  TokenVisualization,
  LogProbVisualization,
  ValueModelSchematic,
  AdvantageCalculationViz,
  PaddingMaskingViz,
  PPOTrainingViz,
  PairwiseRewardDataViz,
  RewardModelTraining,
} from "./pages";

/**
 * App — Minimal Shell (No Top Nav)
 * --------------------------------
 * • Removes nav bar as requested
 * • Adds subtle gradient background + centered max width
 * • Page content sits on soft-card canvas for a modern, "cool" feel
 */

export default function App() {
  return (
    <Router>
      {/* Background gradient */}
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
        {/* Page container */}
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6">
          <div className="rounded-3xl bg-white/90 backdrop-blur shadow-sm ring-1 ring-gray-200">
            {/* Routes */}
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/token-visualization" element={<TokenVisualization />} />
              <Route path="/logprob-visualization" element={<LogProbVisualization />} />
              <Route path="/valuemodel-visualization" element={<ValueModelSchematic />} />
              <Route path="/advantage-visualization" element={<AdvantageCalculationViz />} />
              <Route path="/paddingmasking-visualization" element={<PaddingMaskingViz />} />
              <Route path="/ppo-visualization" element={<PPOTrainingViz />} />
              <Route path="/rewardmodeldatamodeling-visualization" element={<PairwiseRewardDataViz />} />
              <Route path="/rewardmodeltraining" element={<RewardModelTraining />} />
            </Routes>
          </div>
        </div>

        {/* Soft footer badge (optional) */}
        <div className="py-8 text-center text-xs text-gray-400">
          <span className="inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-gradient-to-r from-fuchsia-500 via-sky-500 to-amber-400" />
            RLHF Explained Clearly
          </span>
        </div>
      </div>
    </Router>
  );
}