#include <gtest/gtest.h>
#include "cobali/memory/kv_cache_manager.h"
#include "cobali/common/types.h"
#include "cobali/common/config.h"

using namespace cobali;

TEST(KVCacheManagerTest, AllocateFree) {
    EngineConfig config;
    config.kv_cache_size_mb = 100;
    
    KVCacheManager manager(config);
    manager.initialize(128); // 128 bytes per token
    
    // Allocate
    KVCacheSlot* slot = manager.allocate(1, 2048);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->owner_id, 1);
    EXPECT_TRUE(slot->is_allocated);
    
    size_t used_before = manager.getUsedMemoryBytes();
    EXPECT_GT(used_before, 0);
    
    // Free
    manager.free(1);
    size_t used_after = manager.getUsedMemoryBytes();
    EXPECT_EQ(used_after, 0);
}

TEST(KVCacheManagerTest, MemoryLimit) {
    EngineConfig config;
    config.kv_cache_size_mb = 1; // Very small
    
    KVCacheManager manager(config);
    manager.initialize(128);
    
    // Allocate until out of memory
    KVCacheSlot* slot1 = manager.allocate(1, 1024);
    EXPECT_NE(slot1, nullptr);
    
    // This should fail (not enough memory)
    KVCacheSlot* slot2 = manager.allocate(2, 10000);
    EXPECT_EQ(slot2, nullptr);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

