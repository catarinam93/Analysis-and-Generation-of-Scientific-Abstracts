################################################################
#Advanced Topics in Machine Learning - Group P1_C

#Produced by:
#- Catarina Monteiro up202105279
#- Diogo Mendes up202108102
#- Gonçalo Brochado up202106090
#################################################################

from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, MarianMTModel, MarianTokenizer
import torch


app = Flask(__name__)

#Start Models
#Abs to title - Callidior/bert2bert-base-arxiv-titlegen
'''
tokenizer_callidior = AutoTokenizer.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
model_callidior = AutoModelForSeq2SeqLM.from_pretrained("Callidior/bert2bert-base-arxiv-titlegen")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_callidior = model_callidior.to(device)

#Title to abstract and Enlarge Abstract - Llama 3.2 1B Instruct
model_llama = "meta-llama/Llama-3.2-1B-Instruct"
pipe_llama = pipeline(
    "text-generation",
    model=model_llama,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

#Predict Category - Facebook/bart-large-mnli
classifier_bart = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device=0)

#Translate Portuguese - opus-mt-tc-big-en-pt
model_opus = "Helsinki-NLP/opus-mt-tc-big-en-pt"
tokenizer_opus = MarianTokenizer.from_pretrained(model_opus)
model_opus = MarianMTModel.from_pretrained(model_opus)
model_opus.to(device)
'''
#End Models

#criado com base nos dados do Arxiv

categorias_hier = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    "math.AC": "Commutative Algebra",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Metric Geometry",
    "math.MP": "Mathematical Physics",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CD": "Chaotic Dynamics",
    "nlin.CG": "Cellular Automata and Lattice Gases",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.SI": "Exactly Solvable and Integrable Systems",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.space-ph": "Space Physics",
    "quant-ph": "Quantum Physics",
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory"
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/abstract_to_title', methods=['POST'])
def abstract_to_title():
    texto = request.form.get('texto')
    inputs = tokenizer_callidior(f"""Generate Title: {texto}""", return_tensors="pt").to(device)
    outputs = model_callidior.generate(**inputs)
    title = tokenizer_callidior.batch_decode(outputs, skip_special_tokens=True)
    resposta = f"{title[0]}"
    return render_template('index.html', resposta=resposta, texto = texto)

@app.route('/title_to_abstract', methods=['POST'])
def title_to_abstract():
    texto = request.form.get('texto')
    messages = [
    {"role": "system", "content": "Generate a abstract basead in the title that the user will give you. Answer with scientific knowledge and write with scientific paper style. Just give the abstract, dont repeat the title"},
    {"role": "user", "content": f"Title: {texto}"},
    ]
    outputs = pipe_llama(
        messages,
        max_new_tokens=300,
    )
    resposta = outputs[0]["generated_text"][-1]['content']
    return render_template('index.html', resposta=resposta, texto = texto)

@app.route('/predict_category', methods=['POST'])
def predict_category():
    texto = request.form.get('texto')
    candidate_labels = list(categorias_hier.values())
    label = classifier_bart(texto, candidate_labels, multi_label=True)['labels']
    label = label[:5]
    l = ''
    for i in label:
        s = i + ': ' + i + '\n'
        l += (s)
    resposta = f"{l}"
    return render_template('index.html', resposta=resposta, texto = texto)

@app.route('/translate_portuguese', methods=['POST'])
def translate_portuguese():
    texto = request.form.get('texto')
    src_text = f">>por<< {texto}"
    fragmented_text = [">>por<< " + sentence for sentence in src_text.split(". ")]
    translated_text = ''
    for sentence in fragmented_text:
        inputs = tokenizer_opus(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        translated = model_opus.generate(**inputs, max_length=300)
        translated_text += tokenizer_opus.decode(translated[0], skip_special_tokens=True)
    #inputs = tokenizer_opus(src_text, return_tensors="pt", padding=True).to(device)
    #translated = model_opus.generate(**inputs)
    #r = tokenizer_opus.decode(translated[0], skip_special_tokens=True)
    resposta = f"{translated_text}"
    return render_template('index.html', resposta=resposta, texto = texto)

@app.route('/enlarge_abstract', methods=['POST'])
def enlarge_abstract():
    texto = request.form.get('texto')
    messages = [
    {"role": "system", "content": "Complete the given abstract. Answer with scientific knowledge and write with scientific paper style. Complete until reach the near 900 caracteres"},
    {"role": "user", "content": f"Abstract: {abs}"},
    ]
    outputs = pipe_llama(
        messages,
        max_new_tokens = 300,
    )
    resposta = outputs[0]["generated_text"][-1]['content']
    return render_template('index.html', resposta=resposta, texto = texto)

if __name__ == '__main__':
    app.run(debug=True)
