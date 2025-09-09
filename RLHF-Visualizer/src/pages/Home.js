import React from "react";
import { Link } from "react-router-dom";
import logo from "../assets/vizuara-logo.png"; // adjust path as needed

/**
 * Home — RLHF Process (Minimal with Vizuara Logo)
 * ----------------------------------------------
 * • Removes cluttered nav
 * • Minimal Apple-style typography
 * • Shows process with direct arrows between cards
 * • Includes Vizuara logo prominently at top
 */

const Card = ({ to, title, desc }) => (
  <Link
    to={to}
    className="flex-1 p-5 bg-white rounded-xl shadow-sm ring-1 ring-gray-200 hover:shadow-md transition-shadow"
  >
    <h3 className="text-base font-semibold mb-1">{title}</h3>
    <p className="text-sm text-gray-600">{desc}</p>
  </Link>
);

const Arrow = () => (
  <div className="flex items-center justify-center text-gray-400">
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </svg>
  </div>
);

export default function Home() {
  return (
    <div
      className="max-w-6xl mx-auto p-6 space-y-10"
      style={{ fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif" }}
    >
      {/* Logo */}
      <div className="flex justify-center mb-6">
        <img src={logo} alt="Vizuara AI Labs" className="h-24" />
      </div>

      <header className="text-center">
        <h1 className="text-3xl md:text-4xl font-bold mb-2">RLHF — End-to-End Process</h1>
        <p className="text-gray-500">From Reward Modeling to PPO Training</p>
      </header>

      {/* Reward Modeling */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-center">Reward Modeling</h2>
        <div className="flex flex-col md:flex-row items-center gap-4 justify-center">
          <Card
            to="/rewardmodeldatamodeling-visualization"
            title="Reward Model Data Processing"
            desc="Prepare (prompt, chosen, rejected), pad/mask to block size"
          />
          <Arrow />
          <Card
            to="/rewardmodeltraining"
            title="Reward Model Training"
            desc="Pairwise loss −logσ(Δ) and accuracy visualization"
          />
        </div>
      </section>

      {/* PPO Training */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-center">PPO Training</h2>
        <div className="flex flex-col gap-6">
          <div className="flex flex-col md:flex-row items-center gap-4 justify-center">
            <Card to="/token-visualization" title="Token Shifting" desc="Input–target alignment for LM" />
            <Arrow />
            <Card to="/logprob-visualization" title="Log Probabilities" desc="Token log-probs with teacher forcing" />
            <Arrow />
            <Card to="/valuemodel-visualization" title="Value Model" desc="Predict returns with value head" />
          </div>
          <div className="flex flex-col md:flex-row items-center gap-4 justify-center">
            <Card to="/advantage-visualization" title="Advantage" desc="Compute A_t (GAE)" />
            <Arrow />
            <Card to="/paddingmasking-visualization" title="Padding & Masks" desc="Why padding/masking are needed" />
            <Arrow />
            <Card to="/ppo-visualization" title="PPO Training Loop" desc="Mini-batching, clipped loss, updates" />
          </div>
        </div>
      </section>
    </div>
  );
}