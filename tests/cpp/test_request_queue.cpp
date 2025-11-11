#include <gtest/gtest.h>
#include "cobali/scheduler/request_queue.h"
#include "cobali/common/types.h"

using namespace cobali;

TEST(RequestQueueTest, EnqueueDequeue) {
    RequestQueue queue;
    
    std::vector<Token> tokens = {1, 2, 3};
    Request req1(1, tokens);
    Request req2(2, tokens);
    
    queue.enqueue(&req1);
    queue.enqueue(&req2);
    
    EXPECT_EQ(queue.size(), 2);
    
    Request* dequeued1 = queue.dequeue();
    EXPECT_EQ(dequeued1->id, 1);
    
    Request* dequeued2 = queue.dequeue();
    EXPECT_EQ(dequeued2->id, 2);
    
    EXPECT_TRUE(queue.empty());
}

TEST(RequestQueueTest, Priority) {
    RequestQueue queue;
    
    std::vector<Token> tokens = {1, 2, 3};
    Request req1(1, tokens);
    Request req2(2, tokens);
    Request req3(3, tokens);
    
    req1.priority = Priority::LOW;
    req2.priority = Priority::HIGH;
    req3.priority = Priority::NORMAL;
    
    queue.enqueue(&req1);
    queue.enqueue(&req2);
    queue.enqueue(&req3);
    
    // Should dequeue in priority order: HIGH > NORMAL > LOW
    Request* dequeued1 = queue.dequeue();
    EXPECT_EQ(dequeued1->priority, Priority::HIGH);
    
    Request* dequeued2 = queue.dequeue();
    EXPECT_EQ(dequeued2->priority, Priority::NORMAL);
    
    Request* dequeued3 = queue.dequeue();
    EXPECT_EQ(dequeued3->priority, Priority::LOW);
}

TEST(RequestQueueTest, TryDequeue) {
    RequestQueue queue;
    
    // Empty queue
    Request* req = queue.tryDequeue();
    EXPECT_EQ(req, nullptr);
    
    // Non-empty queue
    std::vector<Token> tokens = {1, 2, 3};
    Request req1(1, tokens);
    queue.enqueue(&req1);
    
    Request* dequeued = queue.tryDequeue();
    EXPECT_NE(dequeued, nullptr);
    EXPECT_EQ(dequeued->id, 1);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

