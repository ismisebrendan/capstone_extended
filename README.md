This contains the code for my 4th year undergratudate capstone project. The first few folders are mostly me experimenting with the initial code and making some changes.

This was initially based off code by Anna Cestaro, a summer undergraduate intern at Armagh Observatory in summer 2024.



# False Positive Detection
This code generates spectra with all expected lines present, spectra with only some lines present and an attempt made to fit lines to the noise, and 'spectra' of noise only with lines fit to that.

This data is then used to train a neural network that attempts to differentiate false and real positives while also not rejecting true positives.


# Recover Parameters
Trying to recover the parameters of an emission line for different A/N and determining at what point they become biased and can no longer be recovered accurately.


# Funcs
The folder containing the relevant functions for this project

# Spectrum Object
The python object responsible for the generation and fitting of all these spectra.

# Old Code
The major steps that led towards the code here, it can be ignored.