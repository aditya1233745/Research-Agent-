# Run: What is quantum entanglement?

*Mode: live (claude-sonnet-4-6 + DuckDuckGo) | Step budget: 8 | Steps used: 8*

## Trace

**Step 1 -- plan:** `search` -- Search for reliable sources explaining quantum entanglement.
  -> search(`quantum entanglement explained`) found sources: S1, S2, S3, S4
**Step 2 -- plan:** `summarise` -- Caltech Science Exchange is a reliable source with a clear explanation; summarising it first.
  -> **summarise** failed: not fetched (handled gracefully)
**Step 3 -- plan:** `fetch` -- Fetching the Caltech Science Exchange page for a clear, accessible explanation of quantum entanglement.
  -> fetch(`S2`) got 5785 chars
**Step 4 -- plan:** `summarise` -- S2 is already fetched, so I should summarise it to get a citable source.
  -> summarise(`S2`): Quantum entanglement is a phenomenon where two or more particles become correlated such that measuring the state of one instantly reveals information about its partner, regardless of the distance between them. This correlation is not due to hidden pre-determined properties within the particles, but arises from the connections between them — particles exist in superposition until measured, at which point their states become correlated. Entanglement does not involve faster-than-light communication; rather, as one researcher notes, "there can be correlation without communication."
**Step 5 -- plan:** `summarise` -- Summarise the already-fetched S2 source to get citable content.
  -> summarise(`S2`): Quantum entanglement is a phenomenon where two or more particles become correlated such that measuring the state of one instantly reveals information about its partner, regardless of the distance separating them. This correlation is not due to hidden information within the particles, but is an intrinsic property arising from the connections between them. Importantly, this does not involve faster-than-light communication — as one researcher notes, "there can be correlation without communication."
**Step 6 -- plan:** `fetch` -- Fetching Wikipedia for a comprehensive source to complement S2.
  -> fetch(`S1`) got 8015 chars
**Step 7 -- plan:** `summarise` -- Summarise the already-fetched Wikipedia article to get a detailed source I can cite alongside S2.
  -> summarise(`S1`): Quantum entanglement is a phenomenon where the quantum state of particles in a group cannot be described independently of each other, even when separated by large distances. Measurements on entangled particles can show perfect correlations — for example, if one particle has clockwise spin, its entangled partner will show counterclockwise spin when measured on the same axis. Unlike classical physics, this correlation cannot be explained by local hidden variables, and while entanglement produces statistical correlations across distances, it cannot be used to transmit information faster than light.
**Step 8 -- plan:** `finish` -- Both S1 and S2 are summarised and provide sufficient detail to construct a thorough answer.
  -> citation check: all claims traceable to sources ✅

## Final Answer

Quantum entanglement is a phenomenon where the quantum state of particles in a group cannot be described independently of each other, even when separated by large distances [S1]. When two or more particles are entangled, measuring the state of one instantly reveals information about its partner, no matter how far apart they are [S2]. These correlations cannot be explained by hidden information stored inside the particles themselves — they are an intrinsic property arising from the connections between them [S2]. For example, if one entangled particle is measured to have a clockwise spin, its partner will always show counterclockwise spin when measured on the same axis [S1]. Unlike classical physics, this behavior cannot be explained by local hidden variables, as confirmed by experiments testing Bell's inequalities [S1]. Crucially, entanglement does not allow faster-than-light communication; as researchers put it, there can be 'correlation without communication' [S2].

Sources:
[S1] Quantum entanglement - Wikipedia -- https://en.wikipedia.org/wiki/Quantum_entanglement
[S2] What Is Quantum Entanglement? Quantum Entanglement Explained in Simple Terms - Caltech Science Exchange -- https://scienceexchange.caltech.edu/topics/quantum-science-explained/entanglement
[S3] What Is the Spooky Science of Quantum Entanglement? -- https://science.nasa.gov/what-is-the-spooky-science-of-quantum-entanglement/  (fetched but not used in final answer)
[S4] What is quantum entanglement? The physics of 'spooky action at a distance' explained | Space -- https://www.space.com/31933-quantum-entanglement-action-at-a-distance.html  (fetched but not used in final answer)

## Errors / non-fatal issues encountered

- summarise requested for [S2] which hasn't been fetched