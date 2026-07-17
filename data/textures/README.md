# Texture per l'avatar SMPL

In questa cartella vanno due tipi di file (non versionati nel repo per
motivi di licenza):

1. **`smpl_uv.obj`** — il template UV ufficiale di SMPL, obbligatorio per
   applicare qualunque texture. Si scarica da <https://smpl.is.tue.mpg.de>
   (sezione Downloads, con lo stesso account usato per i modelli `.pkl`).
   Contiene la mappatura tra i 6890 vertici della mesh SMPL e le coordinate
   dell'immagine texture (con le "cuciture" dove la pelle 3D viene aperta
   sul piano 2D).

2. **Immagini texture** (`.png`/`.jpg`) nel layout UV di SMPL, ad esempio:
   - le texture di esempio incluse nei download di SMPL/SMPL-X di MPI;
   - le texture del dataset SURREAL (<https://www.di.ens.fr/willow/research/surreal/>,
     richiede registrazione accademica): centinaia di texture fotorealistiche
     di corpi vestiti/non vestiti nel layout SMPL.

Una volta copiati qui i file, il menu a tendina **Texture** nella sezione
"Avatar Preset" dell'app elenca automaticamente le immagini trovate.
La texture funziona solo con il Body Model **SMPL** (per SMPL-X non esiste
un template UV pubblico equivalente in questo progetto).

L'export OBJ (`File > Export 3D Mesh (OBJ)`) include coordinate UV, file
`.mtl` e texture, quindi l'avatar testurizzato si può importare direttamente
in Blender/Unity.
