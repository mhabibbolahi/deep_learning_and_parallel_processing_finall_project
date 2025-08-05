Big Data and Parallel Processing Final Project
Docker
To containerize the project, we configured two files: requirements.txt, which lists all the libraries used in the project along with their versions, and Dockerfile, which specifies the setup for the container image and the commands to execute. To deploy the project on a server, you only need to run the build and run commands, and the Dockerized application will be up and running.
FastAPI
The core of the project is main.py, powered by FastAPI, which serves an index.html page to users upon request. This page collects user inputs, such as which project output they want to explore or which specific section of the project. Based on the user's selection, an API request is sent to the server specifying their requirements.
If an image and a network type are provided, the server saves the image and passes it to the selected network model. The output—either a mask or a label—is then sent to the image_result.html page for display. If the user selects a CNN, the input image is returned with its label; if an FCN is chosen, a mask is returned.
If the user opts for parallel processing project outputs, they must specify which part, section, and scenario they want. These selections are sent to the server, which executes the corresponding code, stores the results in a data structure, and passes it back to main.py. The results are then displayed on the result.html page.
Note that multiprocessing code does not require terminal execution as long as the script includes the standard if __name__ == '__main__': block. For this reason, the command to run the core program is embedded within the script to avoid issues with multiprocessing execution.
Big Data
FCN
To implement the FCN model on the Oxford-IIIT Pet dataset, we first normalized the input images using code available in preprocess.py. After normalization, we designed a network (shown below) to train and evaluate the model:

The model's accuracy was evaluated, and the trained model was saved as final_model_fcn.h5. To use this model, the fcn_use_model.py script loads the pretrained model, normalizes the input image, processes it, and generates a mask. The mask is adjusted to highlight differences visible to the human eye, saved as an image, and its path is returned.
CNN
Similar to the FCN approach, but normalization is unnecessary here since the data is already normalized. The input data is split into input and output components, and the model is trained using the network architecture shown below:

The trained model is saved, and a separate script, similar to the FCN one, loads the model, processes the input image, and returns the corresponding label.
Parallel Processing
Part 1: Thread-Based Parallelism
For each exercise, outputs are stored in a global list, which is then passed to the core program and displayed on the results page.
1) Defining a Thread
The code is straightforward and self-explanatory, so we focus on the scenarios:

Scenario 1: Each thread is created, executed, and waits for completion before the next thread starts.
Scenario 2: All threads are created and executed simultaneously, then wait for completion. This is the fastest scenario.
Scenario 3: A hybrid approach where half the threads run sequentially, and the other half run concurrently.

2) Determining the Current Thread
Each function takes a random parameter, prints a start message, pauses based on the parameter, and then prints a completion message. The scenarios are:

Scenario 1: All functions start in parallel, then wait for completion.
Scenario 2: Despite multithreading, functions execute sequentially using join.
Scenario 3: The last two functions run in parallel but sequentially relative to the first.

3) Defining a Thread Subclass
This section mirrors Section 1 but assigns custom names to threads. The key difference is that threads are defined as classes inheriting from threading.Thread, offering greater customization and access to threading tools. The code to be executed is placed in the run method of the class, and threads are started with .start() as before.
4) Thread Synchronization with a Lock
A lock is used to manage access to a critical section where simultaneous execution must be avoided. The first thread to reach the critical section acquires the lock, while others wait until it’s released. The scenarios are:

Scenario 1: All threads are started, then wait for completion.
Scenario 2: Similar to the first, but the lock is applied only to the sleep section, allowing other parts to run in parallel.
Scenario 3: Like the previous scenarios, but the lock is applied to the initial section of the threads.

5) Thread Synchronization with RLock
This involves two functions: one adds and the other removes (based on a random input). A lock ensures that adding and removing don’t happen simultaneously. The scenarios differ in the sleep durations within the locked sections, affecting the output.
6) Thread Synchronization with Semaphores
This implements a producer-consumer problem using semaphores. The producer increments the semaphore after producing, and the consumer decrements it. When the semaphore reaches zero, the consumer waits until the producer increases it again. The scenarios are:

Scenario 1: 10 producers and 10 consumers are created, synchronized with semaphores, and all complete using join.
Scenario 2: All consumers are created first and wait for producers to generate items.
Scenario 3: One producer and one consumer run, complete their process, and then the next pair is created.

7) Thread Synchronization with a Barrier
This simulates a race where a "game over" message is printed once all players reach the finish line. The scenarios are:

Scenario 1: Uses a global variable to track completion without a barrier.
Scenario 2: Uses join to achieve the same result.
Scenario 3: Uses a barrier to synchronize the threads.

Part 2: Process-Based Parallelism
Outputs for each exercise are stored in a queue. After the program completes, the queue’s contents are transferred to a list, passed to the core program, and displayed on the results page.
1) Spawning a Process
This mirrors Section 1 of threading but uses multiprocessing. The function takes an input number and prints lines from 0 to that number (e.g., input 5 prints 0, 1, 2, 3, 4). The scenarios are identical to those in threading.
2) Naming a Process
Processes are assigned custom names, and the name of each process is retrieved during execution. The scenarios are:

Scenario 1: Two processes (one with a custom name, one default) start and wait for completion.
Scenario 2: Processes run sequentially, one after the other.
Scenario 3: The sleep time is removed to minimize execution time.

3) Running Processes in the Background
Background processes don’t produce output directly, so outputs are stored in a queue for display. To differentiate background processes, we exclude their output from the queue if they’re marked as background. The scenarios are:

Scenario 1: One process is background, the other is not; only non-background output is shown.
Scenario 2: Both processes are non-background, but background process output is still excluded.
Scenario 3: Both are background processes, with output excluded as before.

4) Killing a Process
This section explores terminating processes before completion. A function prints numbers 0 to 10 with a one-second delay. The scenarios are:

Scenario 1: The process is killed immediately after starting.
Scenario 2: We wait until the process is no longer alive before killing it.
Scenario 3: The process is given 5 seconds to complete; if it doesn’t, it’s killed.

5) Defining Processes in a Subclass
Processes are defined in a class inheriting from multiprocessing.Process, offering greater flexibility for development. The scenarios are:

Scenario 1: Processes run sequentially.
Scenario 2: Processes run fully in parallel.
Scenario 3: Half the processes run sequentially, half in parallel.

6) Using a Queue to Exchange Data
The producer-consumer problem is implemented using a queue for data exchange. The scenarios are:

Scenario 1: Synchronization is achieved using sleep to avoid issues.
Scenario 2: Semaphores and loops are embedded in the producer and consumer.
Scenario 3: Uses join for synchronization.

7) Synchronizing Processes
The scenarios compare:

Scenario 1: Running two processes with and without a barrier.
Scenario 2: Replacing the barrier with a semaphore, achieving similar results.
Scenario 3: Removing the lock to produce equivalent output.

8) Using a Process Pool
A process pool is used for parallel computations on an array. The scenarios are:

Scenario 1: A pool is created, mapped to a function, and outputs are collected before closing the pool and waiting for completion.
Scenario 2: Uses a with statement for the pool.
Scenario 3: Replaces the square operation with a cube operation.
