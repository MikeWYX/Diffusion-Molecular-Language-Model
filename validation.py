import selfies as sf

benzene_sf = "[C][C][=C][C][=C][Branch2][Branch2][Ring2][C][=C][C][C][C][=Branch1][C][=O][N][Branch1][=Branch2][C][=O][C][=C][C][=C][C][Ring1][C][C][C][C][=Branch1][C][=O][C][=O][C][C][C][Branch1][Branch1][C][=O][C][C][=C][C][=C][C][C][C][C][C][C][C][Ring1][=C][C][C][C][=C][Ring2][Ring1][C]"

# SMILES -> SELFIES -> SMILES translation
try:
    # benzene_sf = sf.encoder(benzene)  # [C][=C][C][=C][C][=C][Ring1][=Branch1]
    benzene_smi = sf.decoder(benzene_sf)  
except sf.DecoderError as e:
    print("Decoding Error:", e)
else:
    print("SELFIES is valid")
    print(benzene_smi)

# symbols_benzene = list(sf.split_selfies(benzene_sf))
# ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[=Branch1]']