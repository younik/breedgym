from gymnasium.envs.registration import register


register(
    id="BreedGym",
    entry_point="breedgym.breedgym:BreedGym",
)

register(
    id="SimplifiedBreedGym",
    entry_point="breedgym.wrappers:SimplifiedBreedGym",
)

register(
    id="KBestBreedGym",
    entry_point="breedgym.wrappers:KBestBreedGym",
)

# ------------ VECTOR ENVS --------------

register(
    id="VecBreedGym",
    entry_point="breedgym.vector.vec_env:VecBreedGym",
)

register(
    id="SelectionScores",
    entry_point="breedgym.vector.vec_wrappers:SelectionScores",
)

register(
    id="PairScores",
    entry_point="breedgym.vector.vec_wrappers:PairScores",
)
