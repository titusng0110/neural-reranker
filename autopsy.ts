import { loadModelWeights, FEATURE_NAMES, NUM_FEATURES, HIDDEN_NEURONS } from './weights_loader.ts';

function silu(x: number): number {
  return x / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function dSilu(x: number): number {
  const sig = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
  return sig * (1 + x * (1 - sig));
}

function drawBar(value: number, maxVal: number, width: number = 24): string {
  const normalized = Math.max(-1, Math.min(1, value / (maxVal || 1e-6)));
  const half = Math.floor(width / 2);
  const chars = Math.round(Math.abs(normalized) * half);

  if (normalized >= 0) {
    return ' '.repeat(half) + '│' + '█'.repeat(chars) + ' '.repeat(half - chars);
  } else {
    return ' '.repeat(half - chars) + '█'.repeat(chars) + '│' + ' '.repeat(half);
  }
}

function runDiagnosis(): void {
  const model = loadModelWeights('./weights.bin');

  console.log('\n=============================================================');
  console.log('🔬 NEURAL NETWORK AUTOPSY: Hybrid SwiGLU Architecture (1028 bytes)');
  console.log('=============================================================');

  // =========================================================================
  // 1. CALIBRATION & BIAS METRICS
  // =========================================================================
  const baseLogit = model.b_out;
  const baseProb = 1 / (1 + Math.exp(-(baseLogit + model.calibrationBias) / model.temperature));

  console.log('\n📊 1. CALIBRATION & BASELINE OPERATING POINT:');
  console.log('-------------------------------------------------------------');
  console.log(`  Raw Output Bias (b_out):        ${baseLogit.toFixed(4)}`);
  console.log(`  Calibration Temperature (T):   ${model.temperature.toFixed(4)}`);
  console.log(`  Calibration Bias:              ${model.calibrationBias.toFixed(4)}`);
  console.log(`  Default Zero-Feature Prob:     ${(baseProb * 100).toFixed(2)}% (confidence when no features match)`);

  // =========================================================================
  // 2. GLOBAL FEATURE SENSITIVITY
  // =========================================================================
  const xOp = new Float32Array(NUM_FEATURES).fill(0.5);

  const gateVals = new Float32Array(HIDDEN_NEURONS);
  const valVals = new Float32Array(HIDDEN_NEURONS);

  for (let j = 0; j < HIDDEN_NEURONS; j++) {
    let g = model.b_gate[j]!;
    let v = model.b_val[j]!;
    for (let i = 0; i < NUM_FEATURES; i++) {
      g += xOp[i]! * model.W_gate[i * HIDDEN_NEURONS + j]!;
      v += xOp[i]! * model.W_val[i * HIDDEN_NEURONS + j]!;
    }
    gateVals[j] = g;
    valVals[j] = v;
  }

  // Analytical gradient: dz/dx_i = W_skip[i] + sum_j W_out[j] * [ dSilu(g_j)*W_gate*v_j + silu(g_j)*W_val ]
  const totalSensitivity = new Float32Array(NUM_FEATURES);
  for (let i = 0; i < NUM_FEATURES; i++) {
    let grad = model.W_skip[i]!;
    for (let j = 0; j < HIDDEN_NEURONS; j++) {
      const g = gateVals[j]!;
      const v = valVals[j]!;
      const wG = model.W_gate[i * HIDDEN_NEURONS + j]!;
      const wV = model.W_val[i * HIDDEN_NEURONS + j]!;

      const dh_dxi = dSilu(g) * wG * v + silu(g) * wV;
      grad += model.W_out[j]! * dh_dxi;
    }
    totalSensitivity[i] = grad;
  }

  const sortedFeatures = Array.from({ length: NUM_FEATURES }, (_, i) => ({
    index: i,
    name: FEATURE_NAMES[i]!,
    skip: model.W_skip[i]!,
    effective: totalSensitivity[i]!
  })).sort((a, b) => Math.abs(b.effective) - Math.abs(a.effective));

  const maxEff = Math.max(...sortedFeatures.map((f) => Math.abs(f.effective))) || 1;

  console.log('\n⚡ 2. GLOBAL FEATURE IMPORTANCE (Total Effective Gradient):');
  console.log('   (Combined Skip Highway + Direct SwiGLU non-linear influence on final logit)');
  console.log('--------------------------------------------------------------------------------------');
  console.log(' Rank | Feature Name                   | Skip   | Total Impact | Visual Strength');
  console.log('--------------------------------------------------------------------------------------');
  sortedFeatures.forEach((f, rank) => {
    const rStr = `#${rank + 1}`.padEnd(5);
    const nameStr = f.name.padEnd(30);
    const skipStr = (f.skip >= 0 ? '+' : '') + f.skip.toFixed(3);
    const effStr = (f.effective >= 0 ? '+' : '') + f.effective.toFixed(3);
    const bar = drawBar(f.effective, maxEff, 24);
    console.log(` ${rStr}| ${nameStr} | ${skipStr.padStart(6)} | ${effStr.padStart(12)} | ${bar}`);
  });

  // =========================================================================
  // 3. THE 6 HIDDEN SWIGLU NEURONS (ROLE DECONSTRUCTION)
  // =========================================================================
  console.log(`\n🧠 3. THE ${HIDDEN_NEURONS} SWIGLU HIDDEN NEURONS (INTERNAL SPECIALIZATION):`);
  console.log('--------------------------------------------------------------------------------------');

  for (let j = 0; j < HIDDEN_NEURONS; j++) {
    const wOut = model.W_out[j]!;
    const neuronRole = wOut > 0 ? '🟢 Excitatory (Boosts Score)' : '🔴 Inhibitory (Suppresses Noise)';

    const featureWeights = Array.from({ length: NUM_FEATURES }, (_, i) => ({
      name: FEATURE_NAMES[i]!,
      gate: model.W_gate[i * HIDDEN_NEURONS + j]!,
      val: model.W_val[i * HIDDEN_NEURONS + j]!
    }));

    const topGate = [...featureWeights].sort((a, b) => Math.abs(b.gate) - Math.abs(a.gate)).slice(0, 2);
    const topVal = [...featureWeights].sort((a, b) => Math.abs(b.val) - Math.abs(a.val)).slice(0, 2);

    console.log(`\nNeuron #${j + 1} [Output Weight W_out = ${(wOut >= 0 ? '+' : '') + wOut.toFixed(4)}] -> ${neuronRole}`);
    console.log(`   Gate Triggered By:  ${topGate.map((t) => `${t.name} (${t.gate.toFixed(2)})`).join(', ')}`);
    console.log(`   Value Scaled By:    ${topVal.map((t) => `${t.name} (${t.val.toFixed(2)})`).join(', ')}`);
  }

  // =========================================================================
  // 4. NON-LINEAR FEATURE SYNERGIES
  // =========================================================================
  interface Interaction {
    featA: string;
    featB: string;
    strength: number;
  }
  const interactions: Interaction[] = [];

  for (let i = 0; i < NUM_FEATURES; i++) {
    for (let k = i + 1; k < NUM_FEATURES; k++) {
      let coupling = 0.0;
      for (let j = 0; j < HIDDEN_NEURONS; j++) {
        const out = model.W_out[j]!;
        const gA = model.W_gate[i * HIDDEN_NEURONS + j]!;
        const vA = model.W_val[i * HIDDEN_NEURONS + j]!;
        const gB = model.W_gate[k * HIDDEN_NEURONS + j]!;
        const vB = model.W_val[k * HIDDEN_NEURONS + j]!;

        coupling += out * (gA * vB + gB * vA);
      }
      interactions.push({
        featA: FEATURE_NAMES[i]!,
        featB: FEATURE_NAMES[k]!,
        strength: coupling
      });
    }
  }

  interactions.sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength));

  console.log('\n\n🔗 4. TOP NON-LINEAR FEATURE SYNERGIES (Cross-Coupling):');
  console.log('   (Pairs of features that mutually amplify or suppress each other in SwiGLU)');
  console.log('--------------------------------------------------------------------------------------');
  interactions.slice(0, 8).forEach((item, idx) => {
    const sign = item.strength > 0 ? '✨ Synergistic (+Boost)' : '⚖️  Compensatory (-Tradeoff)';
    console.log(
      `  #${idx + 1} [${(item.strength >= 0 ? '+' : '') + item.strength.toFixed(3)}] ${item.featA}  ×  ${item.featB}`
    );
    console.log(`     ↳ ${sign}`);
  });

  console.log('\n=============================================================\n');
}

runDiagnosis();
