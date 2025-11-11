#include <gtest/gtest.h>
#include "cobali/scheduler/batch_manager.h"
#include "cobali/common/types.h"
#include "cobali/common/config.h"

using namespace cobali;

TEST(BatchManagerTest, FormBatch) {
    EngineConfig config;
    config.max_batch_size = 4;
    config.max_tokens_per_batch = 1024;
    
    BatchManager manager(config);
    
    RequestQueue pending;
    std::vector<Request*> active;
    
    // Add requests to pending
    std::vector<Token> tokens1(10, 1);
    std::vector<Token> tokens2(20, 2);
    Request req1(1, tokens1);
    Request req2(2, tokens2);
    
    pending.enqueue(&req1);
    pending.enqueue(&req2);
    
    // Form batch
    Batch batch = manager.formBatch(pending, active);
    
    EXPECT_EQ(batch.size(), 2);
    EXPECT_GT(batch.total_tokens, 0);
}

TEST(BatchManagerTest, TokenBudget) {
    EngineConfig config;
    config.max_tokens_per_batch = 100;
    config.prefill_chunk_size = 32;
    
    BatchManager manager(config);
    
    std::vector<Token> tokens(50, 1);
    Request req(1, tokens);
    
    // Prefill phase
    req.phase = Phase::PREFILL;
    int budget = manager.getRequestTokenBudget(&req);
    EXPECT_LE(budget, config.prefill_chunk_size);
    
    // Decode phase
    req.phase = Phase::DECODE;
    budget = manager.getRequestTokenBudget(&req);
    EXPECT_EQ(budget, 1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

