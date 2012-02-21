#include "Buffer.h"

Buffer::Buffer(int length) {
	// malloc length
}

bool Buffer::push(char load) {
	return true;
}

char Buffer::pop() {
	return 'a';
}

bool Buffer::isFull() {
	return head == length;
}

bool Buffer::isEmpty() {
	return head == 0;
}
