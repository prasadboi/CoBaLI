#include <gtest/gtest.h>
#include "cobali/scheduler/prefill_splitter.h"
#include "cobali/common/types.h"
#include "cobali/common/config.h"

using namespace cobali;

TEST(PrefillSplitterTest, ChunkSize) {
    EngineConfig config;
    config.enable_prefill_splitting = true;
    config.prefill_chunk_size = 512;
    config.decode_priority_weight = 0.7f;
    
    PrefillSplitter splitter(config);
    
    std::vector<Token> tokens(1000, 1);
    Request req(1, tokens);
    req.phase = Phase::PREFILL;
    
    // No decode requests
    int chunk = splitter.getChunkSize(&req, 2048, 0);
    EXPECT_EQ(chunk, 512);
    
    // Many decode requests (should reduce chunk)
    chunk = splitter.getChunkSize(&req, 2048, 10);
    EXPECT_LT(chunk, 512);
    EXPECT_GT(chunk, 0);
}

TEST(PrefillSplitterTest, ShouldSplit) {
    EngineConfig config;
    config.enable_prefill_splitting = true;
    config.prefill_chunk_size = 512;
    
    PrefillSplitter splitter(config);
    
    // Large prompt - should split
    std::vector<Token> large_tokens(1000, 1);
    Request large_req(1, large_tokens);
    EXPECT_TRUE(splitter.shouldSplit(&large_req));
    
    // Small prompt - no split needed
    std::vector<Token> small_tokens(100, 1);
    Request small_req(2, small_tokens);
    EXPECT_FALSE(splitter.shouldSplit(&small_req));
}

TEST(PrefillSplitterTest, UpdateAfterChunk) {
    EngineConfig config;
    config.enable_prefill_splitting = true;
    
    PrefillSplitter splitter(config);
    
    std::vector<Token> tokens(1000, 1);
    Request req(1, tokens);
    req.phase = Phase::PREFILL;
    req.tokens_processed = 0;
    
    // Process first chunk
    splitter.updateAfterChunk(&req, 512);
    EXPECT_EQ(req.tokens_processed, 512);
    EXPECT_EQ(req.phase, Phase::PREFILL);
    
    // Process remaining tokens
    splitter.updateAfterChunk(&req, 488);
    EXPECT_EQ(req.tokens_processed, 1000);
    EXPECT_EQ(req.phase, Phase::DECODE);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

