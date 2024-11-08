#ifndef _WAITER_H_
#define _WAITER_H_
 
#include "common_include.h"
 
namespace vision {
namespace ops {
    
class Waiter {
public:
    explicit Waiter(uint16_t waitCount) : mWaitCount(waitCount), mFinishedCount(0), mLock(), mCond() {}
 
    ~Waiter() = default;
 
    inline void WaitAll()
    {
        std::unique_lock<std::mutex> lk(mLock);
        mCond.wait(lk, [this] { return mFinishedCount == mWaitCount; });
    }
 
    inline void FinishedOne()
    {
        bool notify = false;
        mLock.lock();
        ++mFinishedCount;
        notify = (mFinishedCount == mWaitCount);
        mLock.unlock();
 
        if (notify) {
            mCond.notify_all();
        }
    }
 
private:
    uint16_t mWaitCount = 0;
    uint16_t mFinishedCount = 0;
    std::mutex mLock;
    std::condition_variable mCond;
};
}
}
#endif // _WAITER_H_
