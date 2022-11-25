/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/text/decoder/LexiconFreeSeq2SeqDecoder.h"
#include "flashlight/lib/text/decoder/lm/ZeroLM.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using namespace fl::lib::text;

TEST(Seq2SeqDecoderTest, LexiconFreeBasic) {
  const int T = 3;
  const int N = 4;
  std::vector<float> emissions(T * N);
  std::iota(emissions.begin(), emissions.end(), -(T * N) / 2);
  ASSERT_EQ(emissions.size(), T * N);

  const int eosIdx = 4;
  const int maxOutputLength = 3;

  // Deterministic map from input token idx prediction to output scores.
  // Score geneneration is considered a martingale (i.e. not dependent on
  // previous timestep) for the purposes of testing
  std::unordered_map<int, std::vector<float>> modelScoreMapping = {
      {0, {0.1, 0.1, 0.5, 0.1}},
      {1, {0.5, 0.2, 0.2, 0.1}},
      {2, {0.1, 0.5, 0.1, 0.1}},
  };
  ASSERT_EQ(modelScoreMapping.size(), T);

  // A simulation of model state. These are synthetically created for the test
  // but store information about model scores for the next timestep (which would
  // normally be hidden states
  struct ModelState {
    int timestep; // timestep for this model state
    int tokenIdx; // input token index that produced this model state
    float score; // score of the token emitted at this timestep

    static std::shared_ptr<ModelState>
    create(int timestep, int tokenIdx, float score) {
      auto s = std::make_shared<ModelState>();
      s->timestep = timestep;
      s->tokenIdx = tokenIdx;
      s->score = score;
      return s;
    }
  };

  EmittingModelUpdateFunc updateFunc =
      [&_emissions = emissions, _T = T, _N = N, &modelScoreMapping](
          const float* emissions,
          const int N,
          const int T,
          const std::vector<int>& prevStepTokenIdxs,
          const std::vector<EmittingModelStatePtr>& prevStepModelStates,
          const int& timestep)
      -> std::pair<
          std::vector<std::vector<float>>, // output probs (beamSize x N)
          std::vector<EmittingModelStatePtr> // future beam state
          > {
    std::cout << "update func called with "
              << " N = " << N << " T = " << T << " prevStepTokenIdxs {";
    for (int i : prevStepTokenIdxs) {
      std::cout << i << ", ";
    }
    std::cout << "} "
              << " prevStepModelStates vec of size "
              << prevStepModelStates.size() << " timestep = " << timestep
              << std::endl;

    // Can't use gtest in this lambda since it might generate an empty return
    assert(_emissions.data() == emissions); // Should point to the same data
    assert(_N == N);
    assert(_T == T);
    assert(prevStepTokenIdxs.size() == prevStepModelStates.size());
    if (timestep == 0) {
      // Initial token index is -1 at the first timestep
      assert(prevStepTokenIdxs == std::vector<int>{-1});
      // Timestep 0 has prevStepModelStates == {nullptr}
      assert(prevStepModelStates.size() == 1);
      assert(prevStepModelStates.front() == nullptr);
    } else {
      // Check proper model state propagation and ordering from prev timestep
      // for (modelScoreMapping[t - 1]
      for (size_t i = 0; i < prevStepModelStates.size(); ++i) {
        auto p = std::static_pointer_cast<ModelState>(prevStepModelStates[i]);
        // TODO: check stuff
      }
    }

    auto& curModelScore = modelScoreMapping[timestep];

    // Create model states from the token indices and timesteps
    std::vector<EmittingModelStatePtr> modelStates;
    for (size_t n = 0; n < prevStepTokenIdxs.size(); ++n) {
      if (timestep == 0) {
        modelStates.emplace_back(
            ModelState::create(timestep, n, /* score = */ -1.)); // dummy state
      } else {
        modelStates.emplace_back(
            ModelState::create(timestep, n, /* score = */ curModelScore[n]));
      }
    }

    // Pretend token probabilities are the same for each token in the beam
    std::vector<std::vector<float>> outProbs(
        prevStepTokenIdxs.size(), curModelScore);

    // for (int prevStepTokenIdx : prevStepTokenIdxs) {
    // };

    return {outProbs, modelStates};
  };

  LexiconFreeSeq2SeqDecoderOptions options = {
      .beamSize = 2,
      .beamSizeToken = 4,
      .beamThreshold = 1000,
      .lmWeight = 0, // use ZeroLM
      .eosScore = 0,
      .logAdd = true};

  LexiconFreeSeq2SeqDecoder decoder(
      std::move(options),
      std::make_shared<ZeroLM>(),
      eosIdx,
      std::move(updateFunc),
      maxOutputLength);

  decoder.decodeStep(emissions.data(), T, N);

  std::vector<DecodeResult> hyps = decoder.getAllFinalHypothesis();
  ASSERT_EQ(hyps.size(), options.beamSize);
  ASSERT_FLOAT_EQ(hyps[0].score, 0.5 + 0.5 + 0.5);
  ASSERT_FLOAT_EQ(hyps[1].score, 0.5 + 0.2 + 0.5);

  for (auto& hyp : hyps) {
    ASSERT_EQ(hyp.lmScore, 0); // since using ZeroLM
    std::cout << "DecodeResult {score = " << hyp.score
              << " emittingModelScore = " << hyp.emittingModelScore
              << " lmScore = " << hyp.lmScore
              << " wordsSize = " << hyp.words.size() << " tokensSize "
              << hyp.tokens.size() << "}" << std::endl;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
