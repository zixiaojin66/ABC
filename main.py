from net import *
from smiles2vector import *
import networkx as nx
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from translate import Translator
import requests
import os

# 获取当前文件所在的文件夹路径
current_path = os.path.dirname(os.path.abspath(__file__))
mapping={0:'未发现',1:'非常罕见',2:'罕见',3:'不频繁',4:'频繁',5:'非常频繁'}
f_p=os.path.join(current_path, 'data', 'frequencyMat.csv')
frequencyMat=np.loadtxt(f_p,delimiter=',',dtype='int')
side_effect_label = os.path.join(current_path,'data','side_effect_label_750.mat')
input_dim = 109
cuda_name='cuda:0'
#加载模型
model=GAT3().to(device=cuda_name)
model_p=os.path.join(current_path,'models','net_params.pth')
model.load_state_dict(torch.load(model_p))
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
# print('Device: ', device)


DF=False
not_FC=False
knn=5
pca=False
metric='cosine'

frequencyMat = frequencyMat.T
if pca:
    pca_ = PCA(n_components=256)
    similarity_pca = pca_.fit_transform(frequencyMat)
    print('PCA 信息保留比例： ')
    print(sum(pca_.explained_variance_ratio_))
    A = kneighbors_graph(similarity_pca, knn, mode='connectivity', metric=metric, include_self=False)
else:
    A = kneighbors_graph(frequencyMat, knn, mode='connectivity', metric=metric, include_self=False)
G = nx.from_numpy_matrix(A.todense())
edges = []
for (u, v) in G.edges():
    edges.append([u, v])
    edges.append([v, u])
edges = np.array(edges).T
edges = torch.tensor(edges, dtype=torch.long)
node_label = scipy.io.loadmat(side_effect_label)['node_label']
feat = torch.tensor(node_label, dtype=torch.float)
sideEffectsGraph = Data(x=feat, edge_index=edges)

#预测函数
def predict(model, device,x,edge_index,batch, sideEffectsGraph, DF, not_FC):
    model.eval()
    torch.cuda.manual_seed(42)
    print('Make prediction for {} samples...'.format(1))
    # 对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with torch.no_grad():来强制之后的内容不进行计算图构建
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        pred, _, _ = model(x,edge_index,batch, sideEffectsGraph, DF, not_FC)
    return pred

#找到副作用的顺序
path=os.path.join(current_path,'data','Supplementary Data 1.txt')

fr=open(path,'r')
all_lines=fr.readlines()
dataset=[]
for line in all_lines:
    line=line.strip().split('\t')
    dataset.append(line)
df=pd.DataFrame(dataset[1:],columns=['drug','sideeffect','rate'])
data=list(set(list(df['sideeffect'])))
data.sort()

sideEffects_chinese=pd.read_excel('1.xlsx')
data=list(sideEffects_chinese.iloc[:,1])

class LanguageTrans():
    def __init__(self, mode):
        self.mode = mode
        if self.mode == "E2C":
            self.translator = Translator(from_lang="english", to_lang="chinese")
        if self.mode == "C2E":
            self.translator = Translator(from_lang="chinese", to_lang="english")
    def trans(self, word):
        translation = self.translator.translate(word)
        return translation
#中译英
def translate_name(chinese):
    '''
    返回药物英文名字
    :param chinese:药物中文名
    :return: 药物英文名
    '''
    if is_chinese(chinese):
        translator = LanguageTrans("C2E")
        word = translator.trans(chinese)
    else:
        word=chinese
    return word



def find_Molecular_formula(Eng_name):
    drug_name = Eng_name

    # PubChem API 的基本 URL
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'

    # 发送 HTTP GET 请求并获取响应
    response = requests.get(base_url + drug_name + '/property/MolecularFormula/JSON')

    # 输出响应内容
    return response.json()['PropertyTable']['Properties'][0]['MolecularFormula']

def get_Smiles(Eng_name):
    name = Eng_name
    api_key = 'your_api_key'
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON'
    response = requests.get(url, headers={'API_Key': api_key})

    # 解析响应
    if response.status_code == 200:
        data = response.json()
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return smiles
    else:
        return None
#判断输入是否为汉字：
def is_chinese(text):
    """
    判断一个字符串是否全部由中文字符组成
    """
    for char in text:
        if not '\u4e00' <= char <= '\u9fa5':
            return False
    return True
#举例子
# smile_graph = convert2graph(['C[N+](C)(C)CC(=O)[O-]'])
# x=torch.FloatTensor(smile_graph['C[N+](C)(C)CC(=O)[O-]'][1])
# edge_index=torch.LongTensor(smile_graph['C[N+](C)(C)CC(=O)[O-]'][2])
# batch=[0 for i in range(len(x))]
# batch=torch.LongTensor(batch)
# a=predict(model,device,x,edge_index,batch,sideEffectsGraph,DF,not_FC)


