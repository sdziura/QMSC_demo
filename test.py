from HMMGen import GenHMM

observables = 3
hiddenStates = 5

hid, obs = GenHMM(100, observables=observables, hiddenStates=range(hiddenStates))

print("Hidden chain length: ", len(hid))
for o in range(observables):
    print(f"Observable {o} chain length: ", len(obs[o]))

print("\nHidden chain:\n", hid)
for o in range(observables):
    print("Observable ", o, "\n", obs[o])
