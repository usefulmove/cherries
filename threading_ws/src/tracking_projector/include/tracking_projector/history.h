#ifndef HISTORY_H
#define HISTORY_H


class History
{

public:
    History();
    void add(int count);
    int predict(int nSteps);

private:
    int last;
    int data[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

#endif  // HISTORY_H