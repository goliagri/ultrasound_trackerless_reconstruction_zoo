

##Style

Verifiability and quick identification of when things go wrong is a high priority. 
    Use type-checking in function inputs and outputs
    Liberally check state and raise informative errors if not as expected. Practically every complex function should include these checks to verify inputs and outputs. 
    Keep unit-testability in mind when writing code. It is highly beneficial for functionality to be easily covered by unit tests.
    Include clear documentation of input and output requirements in non-trivial functions
    When writing a for loop or a while loop where relevant variables meaningfully chnage between iterations, if it is sufficiently complex as to not be easily understandable with a 1 minute skim, write an invariant comment above, e.g. #LOOP INVARIANT: (expected state of variables in each iteration of the loop)


