#include "history.h"

History::History() {};

void History::add(int count) {
    last = (last +1)%10;
    data[last] = count;
};

int History::predict(int nSteps) {
    int first = (last +1)%10;
    double avg_delta = (data[last] - data[first])/10.0;
    // the duration timeout is twice of the rate we expect.
    // so if we have 1 timeout it is equivalent to 2 steps 
    int pred_count = int(data[last] + avg_delta * 2 * nSteps);
    // add(pred_count);
    return pred_count;
};