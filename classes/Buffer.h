#ifndef BUFFER_H
#define BUFFER_H

class Buffer {
	private:
		int length, head;

	public:
		Buffer(int length);
		
		bool push(char load);
		char pop();
		bool isFull();
		bool isEmpty();
};

#endif
