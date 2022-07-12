import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os
import MDAnalysis
from MDAnalysis import transformations

def initProteins():
    proteins = pd.DataFrame(columns=['eps_factor','pH','ionic','fasta'])
    fasta_A1 = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_M20GP20S = """GSMASASSSQRSRSGSGNFSGSRSGSFSGNDNFGRSGNFSGRSGFGGSRSGGGYSGSGDSYNSFGNDGSNFSGSGSYNDF
GNYNNQSSNFGPMKSGNFGGRSSGSSGGSGQYFAKPRNQGSYSGSSSSSSYGSSRRF""".replace('\n', '')
    fasta_M30GP30S = """GSMASASSSQRSRSSSGNFSGSRSGSFSGNDNFGRSGNFSGRSGFSGSRSGSGYSGSSDSYNSFGNDSSNFSGSSSYNDF
GNYNNQSSNFGPMKSGNFSGRSSSSSGSSGQYFAKPRNQGSYSGSSSSSSYSSSRRF""".replace('\n', '')
    fasta_M14NP14Q = """GSMASASSSQRGRSGSGQFGGGRGGGFGGQDQFGRGGQFSGRGGFGGSRGGGGYGGSGDGYQGFGQDGSQFGGGGSYQDF
GQYQQQSSQFGPMKGGQFGGRSSGGSGGGGQYFAKPRQQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_M23SP23T = """GSMATATTTQRGRTGTGNFGGGRGGGFGGNDNFGRGGNFTGRGGFGGTRGGGGYGGTGDGYNGFGNDGTNFGGGGTYNDF
GNYNNQTTNFGPMKGGNFGGRTTGGTGGGGQYFAKPRNQGGYGGTTTTTTYGTGRRF""".replace('\n', '')
    fasta_P12D = """GSMASADSSQRDRDDSGNFGDGRGGGFGGNDNFGRGGNFSDRGGFGGSRGDGGYGGDGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFDPMKGGNFGDRSSGPYDGGGQYFAKPRNQGGYGGSSSSSSYGSDRRF""".replace('\n', '')
    fasta_P7R = """GSMASASSSQRGRSGRGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGRYGGSGDRYNGFGNDGRNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFRGRSSGPYGRGGQYFAKPRNQGGYGGSSSSRSYGSGRRF""".replace('\n', '')
    fasta_M4D = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNGNFGRGGNFSGRGGFGGSRGGGGYGGSGGGYNGFGNSGSNFGGGGSYNGFGNYNNQ
SSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P4D = """GSMASASSSQRDRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGDFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSDPYGGGGQYFAKPRNQGGYGGSSSSSSYDSGRRF""".replace('\n', '')
    fasta_P8D = """GSMASASSSQRDRSGSGNFGGGRDGGFGGNDNFGRGDNFSGRGDFGGSRDGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSDPYGGGGQYFAKPRNQDGYGGSSSSSSYDSGRRF""".replace('\n', '')
    fasta_M6R = """GSMASASSSQGGRSGSGNFGGGRGGGFGGNDNFGGGGNFSGSGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGSSSGPYGGGGQYFAKPGNQGGYGGSSSSSSYGSGGRF""".replace('\n', '')
    fasta_M8FP4Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNGGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNYNNQ
SSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P7FM7Y = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGFGGSGDGFNGFGNDGSNFGGGGSFNDFGNFNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQFFAKPRNQGGFGGSSSSSSFGSGRRF""".replace('\n', '')
    fasta_M10R = """GSMASASSSQGGSSGSGNFGGGGGGGFGGNDNFGGGGNFSGSGGFGGSGGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGSSSGPYGGGGQYFAKPGNQGGYGGSSSSSSYGSGGGF""".replace('\n', '')
    fasta_P2R = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFRNDGSNFGGGGRYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P7KP12D = """GSMASADSSQRDRDDKGNFGDGRGGGFGGNDNFGRGGNFSDRGGFGGSRGDGKYGGDGDKYNGFGNDGKNFGGGGSYNDFGNYNNQ
SSNFDPMKGGNFKDRSSGPYDKGGQYFAKPRNQGGYGGSSSSKSYGSDRRF""".replace('\n', '')
    fasta_M3RP3K = """GSMASASSSQRGKSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSKGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRKF""".replace('\n', '')
    fasta_M12FP12Y = """GSMASASSSQRGRSGSGNYGGGRGGGYGGNDNYGRGGNYSGRGGYGGSRGGGGYGGSGDGYNGYGNDGSNYGGGGSYNDYGNYNNQ
SSNYGPMKGGNYGGRSSGGSGGGGQYYAKPRNQGGYGGSSSSSSYGSGRRY""".replace('\n', '')
    fasta_M9FP3Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNGGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNGNNQ
SSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRS""".replace('\n', '')
    fasta_A1S = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGNYNNQ
SSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '') 
    fasta_LAF1 = """MESNQSNNGGSGNAALNRGGRYVPPHLRGGDGGAAAAASAGGDDRRGGAGGGGYRRGGGNSGGGGGGG
YDRGYNDNRDDRDNRGGSGGYGRDRNYEDRGYNGGGGGGGNRGYNNNRGGGGGGYNRQDRGDGGSSNFSRGG
YNNRDEGSDNRGSGRSYNNDRRDNGGDGLEHHHHHH""".replace('\n', '')
    fasta_LAF1D2130 = """MESNQSNNGGSGNAALNRGGDGGAAAAASAGGDDRRGGAGGGGYRRGGGNSGGGGGGGYDRGYNDNRDDR
DNRGGSGGYGRDRNYEDRGYNGGGGGGGNRGYNNNRGGGGGGYNRQDRGDGGSSNFSRGGYNNRDEGSDN
RGSGRSYNNDRRDNGGDGLEHHHHHH""".replace('\n', '')
    fasta_LAF1shuf = """MNNSGDNDRGSGNYGLRNSFGDDGYGDNGNDEGNSGYRNRGLGGDRADEYGNSGGNGDNEAAPNASDRDD
AHYYDSDDYDDGGGGRGSGGAGGGGARGPGSNRAGRYGGGGRRGRGRGNGYNGNRSQRRRGGGRGRGNRG
YRVGNGNGQSGGRNSRGGGGGNGGANYGLEHHHHHH""".replace('\n', '')
    fasta_M14NM4QP18G = """GSMASASSSGRGRSGSGGFGGGRGGGFGGGDGFGRGGGFSGRGGFGGSRGGGGYGGSGDGYGGFGGDGSG
FGGGGSYGDFGGYGGGSSGFGPMKGGGFGGRSSGGSGGGGGYFAKPRGGGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_M10GP10S = """GSMASASSSQRSRSGSGNFGGGRSGGFGGNDNFGRSGNFSGRGGFGGSRGGGGYGGSGDSYNGFGNDGSN
FGGSGSYNDFGNYNNQSSNFGPMKSGNFGGRSSGSSGGSGQYFAKPRNQGSYSGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P23GM23SM12FP12Y = """GSMAGAGGGQRGRGGGGNYGGGRGGGYGGNDNYGRGGNYGGRGGYGGGRGGGGYGGGGDGYNGYGN
DGGNYGGGGGYNDYGNYNNQGGNYGPMKGGNYGGRGGGGGGGGGQYYAKPRNQGGYGGGGGGGGYGGGRRY""".replace('\n', '')
    fasta_P23GM23SP7FM7Y = """GSMAGAGGGQRGRGGGGNFGGGRGGGFGGNDNFGRGGNFGGRGGFGGGRGGGGFGGGGDGFNGFGN
DGGNFGGGGGFNDFGNFNNQGGNFGPMKGGNFGGRGGGGGGGGGQFFAKPRNQGGFGGGGGGGGFGGGRRF""".replace('\n', '')
    fasta_P23GM23S = """GSMAGAGGGQRGRGGGGNFGGGRGGGFGGNDNFGRGGNFGGRGGFGGGRGGGGYGGGGDGYNGFGN
DGGNFGGGGGYNDFGNYNNQGGNFGPMKGGNFGGRGGGGGGGGGQYFAKPRNQGGYGGGGGGGGYGGGRRF""".replace('\n', '')
    fasta_A2 = """GHMGRGGNFGFGDSRGGGGNFGPGPGSNFRGGSDGYGSGRGFGDGYNGYGGGPGG
GNFGGSPGYGGGRGGYGGGGPGYGNQGGGYGGGYDNYGGGNYGSGNYNDFGNYNQQPSNYGPMKSGNFGGSRNMGG
PYGGGNYGPGGSGGSGGYGGRSRY""".replace('\n', '')
    fasta_FUS = """MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYG
QSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQS
SSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS""".replace('\n', '')
    fasta_Ddx4WT = """MGDEDWEAEINPHMSSYVPIFEKDRYSGENGDNFNRTPASSSEMDDGPSR
RDHFMKSGFASGRNFGNRDAGECNKRDNTSTMGGFGVGKSFGNRGFSNSR
FEDGDSSGFWRESSNDCEDNPTRNRGFSKRGGYRDGNNSEASGPYRRGGR
GSFRGCRGGFGLGSPNNDLDPDECMQRTGGLFGSRRPVLSGTGNGDTSQS
RSGSGSERGGYKGLNEEVITGSGKNSWKSEAEGGES""".replace('\n', '')
    fasta_Lge1 = """HHHHHHDYDIPTTENLYFQGAMGSMSGYTGNNYSRYSSTPPRQRGGYHHARRSRGGAGGSYYRGGNASYGARYNSDYEQPPQEG
DLRQTGAYYRNGYTDTRPYYAAWSHPQFEK""".replace('\n', '')
    fasta_Lge1RK = """HHHHHHDYDIPTTENLYFQGAMGSMSGYTGNNYSKYSSTPPKQKGGYHHAKKSKGGAGGSYYKGGNASYGAKYNSDYEQPPQEG
DLKQTGAYYKNGYTDTKPYYAAWSHPQFEK""".replace('\n', '')
    fasta_Lge1YA = """HHHHHHDYDIPTTENLYFQGAMGSMSGATGNNASRASSTPPRQRGGAHHARRSRGGAGGSAARGGNASAGARANSDAEQPPQEG
DLRQTGAAARNGATDTRPAAAAWSHPQFEK""".replace('\n', '')

    proteins.loc['Lge1'] = dict(eps_factor=0.2,pH=7.5,fasta=list(fasta_Lge1),ionic=0.1)
    proteins.loc['Lge1RK'] = dict(eps_factor=0.2,pH=7.5,fasta=list(fasta_Lge1RK),ionic=0.1)
    proteins.loc['Lge1YA'] = dict(eps_factor=0.2,pH=7.5,fasta=list(fasta_Lge1YA),ionic=0.1)
    proteins.loc['Ddx4WT'] = dict(eps_factor=0.2,pH=6.5,fasta=list(fasta_Ddx4WT),ionic=0.13)
    proteins.loc['FUS'] = dict(eps_factor=0.2,pH=5.5,fasta=list(fasta_FUS),ionic=0.15)
    proteins.loc['A2'] = dict(eps_factor=0.2,pH=5.5,fasta=list(fasta_A2),ionic=0.010)
    proteins.loc['P23GM23SM12FP12Y'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P23GM23SM12FP12Y),ionic=0.15)
    proteins.loc['P23GM23SP7FM7Y'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P23GM23SP7FM7Y),ionic=0.15)
    proteins.loc['P23GM23S'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P23GM23S),ionic=0.15)
    proteins.loc['M14NM4QP18G'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M14NM4QP18G),ionic=0.15)
    proteins.loc['M10GP10S'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M10GP10S),ionic=0.15)
    proteins.loc['LAF1'] = dict(eps_factor=0.2,pH=7.5,fasta=list(fasta_LAF1),ionic=0.15)
    proteins.loc['LAF1D2130'] = dict(eps_factor=0.2,pH=7.5,fasta=list(fasta_LAF1D2130),ionic=0.15)
    proteins.loc['LAF1shuf'] = dict(eps_factor=0.2,pH=7.5,fasta=list(fasta_LAF1shuf),ionic=0.15)
    proteins.loc['A1S150'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_A1S),ionic=0.15)
    proteins.loc['A1S200'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_A1S),ionic=0.2)
    proteins.loc['A1S300'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_A1S),ionic=0.3)
    proteins.loc['A1S500'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_A1S),ionic=0.5)
    proteins.loc['M20GP20S'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M20GP20S),ionic=0.15)
    proteins.loc['M30GP30S'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M30GP30S),ionic=0.15)
    proteins.loc['M14NP14Q'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M14NP14Q),ionic=0.15)
    proteins.loc['M23SP23T'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M23SP23T),ionic=0.15)
    proteins.loc['A1'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_A1),ionic=0.15)
    proteins.loc['M12FP12Y'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M12FP12Y),ionic=0.15)
    proteins.loc['P7FM7Y'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P7FM7Y),ionic=0.15)
    proteins.loc['M8FP4Y'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M8FP4Y),ionic=0.15)
    proteins.loc['M9FP3Y'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M9FP3Y),ionic=0.15)
    proteins.loc['M6R'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M6R),ionic=0.15)
    proteins.loc['P4D'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P4D),ionic=0.15)
    proteins.loc['P2R'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P2R),ionic=0.15)
    proteins.loc['P8D'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P8D),ionic=0.15)
    proteins.loc['P7KP12D'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P7KP12D),ionic=0.15)
    proteins.loc['M3RP3K'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M3RP3K),ionic=0.15)
    proteins.loc['M4D'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M4D),ionic=0.15)
    proteins.loc['P12D'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P12D),ionic=0.15)
    proteins.loc['M10R'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_M10R),ionic=0.15)
    proteins.loc['P7R'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_P7R),ionic=0.15)
    return proteins

def genParamsLJ(df,name,prot):
    fasta = prot.fasta.copy()
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a,'MW'] for a in types]
    lj_eps = prot.eps_factor*4.184
    return lj_eps, fasta, types, MWs

def genParamsDH(df,name,prot,temp):
    kT = 8.3145*temp*1e-3
    fasta = prot.fasta.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[prot.fasta[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[prot.fasta[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)
    return yukawa_eps, yukawa_kappa

def genDCD(residues,name,prot,temp,n_chains):
    path = '{:s}/{:d}'.format(name,temp)
    top = md.Topology()
    for _ in range(n_chains):
        chain = top.add_chain()
        for resname in prot.fasta:
            residue = top.add_residue(residues.loc[resname,'three'], chain)
            top.add_atom(residues.loc[resname,'three'], 
                         element=md.element.carbon, residue=residue)
        for i in range(chain.n_atoms-1):
            top.add_bond(chain.atom(i),chain.atom(i+1))

    t = md.load(path+'/{:s}.dcd'.format(name),top=top)
    t.xyz *= 10
    t.unitcell_lengths *= 10
    lz = t.unitcell_lengths[0,2]
    edges = np.arange(-lz/2.,lz/2.,1)
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    zmid = np.apply_along_axis(lambda a: z[a.argmax()], 1, h)
    indices = np.argmin(np.abs(t.xyz[:,:,2]-zmid[:,np.newaxis]),axis=1)
    t[0].save_pdb(path+'/top.pdb')
    t.save_dcd(path+'/traj4.dcd')

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj4.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj3.dcd', ag.n_atoms) as W:
        for ts,ndx in zip(u.trajectory,indices): 
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.center_in_box(
                u.select_atoms('index {:d}'.format(ndx)), center='geometry')(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj3.dcd',top=path+'/top.pdb')
    edges = np.arange(0,lz,1)
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    h = np.mean(h[:120],axis=0)
    maxoverlap = np.apply_along_axis(lambda a: np.correlate(h,np.histogram(a,
                bins=edges)[0], 'full').argmax()-h.size+dz, 1, t.xyz[:,:,2])

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj3.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj2.dcd', ag.n_atoms) as W:
        for ts,mo in zip(u.trajectory,maxoverlap):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.translate([0,0,mo*10])(ts)  
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj2.dcd',top=path+'/top.pdb')
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    zmid = np.apply_along_axis(lambda a: z[a>np.quantile(a,.98)].mean(), 1, h)
    indices = np.argmin(np.abs(t.xyz[:,:,2]-zmid[:,np.newaxis]),axis=1)

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj2.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj1.dcd', ag.n_atoms) as W:
        for ts,ndx in zip(u.trajectory,indices): 
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.center_in_box(
                u.select_atoms('index {:d}'.format(ndx)), center='geometry')(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj1.dcd',top=path+'/top.pdb')
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    h = np.mean(h[120:],axis=0)
    maxoverlap = np.apply_along_axis(lambda a: np.correlate(h,np.histogram(a,
                bins=edges)[0], 'full').argmax()-h.size+dz, 1, t.xyz[:,:,2])

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj1.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj.dcd', ag.n_atoms) as W:
        for ts,mo in zip(u.trajectory,maxoverlap):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.translate([0,0,mo*10])(ts)  
            ts = transformations.wrap(ag)(ts)
            W.write(ag)
   
    t = md.load(path+'/traj.dcd',top=path+'/top.pdb')

    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    np.save('{:s}_{:d}.npy'.format(name,temp),h,allow_pickle=False)
    os.remove(path+'/traj1.dcd')
    os.remove(path+'/traj2.dcd')
    os.remove(path+'/traj3.dcd')
    os.remove(path+'/traj4.dcd')
    t.xyz /= 10
    t.unitcell_lengths /= 10
    t[0].save_pdb(path+'/top.pdb')
    t.save_dcd(path+'/traj.dcd')
