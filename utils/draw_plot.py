import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os,sys,torch, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.compute_metric import filter_data, read_res
from models.tokenizer import Entokenizer, Trajtokenizer
from visualizer import get_local
get_local.activate()
from train import TrajPredTransformerV1, TrajPredTransformerV2,TrajPredTransformerV3, random_seed, load_ETCEn_dataV1, read_config,load_trajdata,load_trajdataV2
from utils.metrics import unnormalize, MinMaxUnnormalize, time_station_Accuracy
from utils.standardization import exp_unnormalize
from bertviz import head_view, model_view
from bertviz.head_view import *
from bertviz.neuron_view import show
from tqdm import tqdm
import seaborn as sns

def stationAttenView(
        attention=None,
        predict_sections_tokens=None,
        true_sections_tokens = None,
        prettify_tokens=True,
        layer=None,
        heads=None,
        encoder_attention=None,
        decoder_attention=None,
        cross_attention=None,
        encoder_tokens=None,
        decoder_tokens=None,
        include_layers=None,
        html_action='view'
):

    attn_data = []
    if attention is not None:
        if predict_sections_tokens is None:
            raise ValueError("'tokens' is required")
        if encoder_attention is not None or decoder_attention is not None or cross_attention is not None \
                or encoder_tokens is not None or decoder_tokens is not None:
            raise ValueError("If you specify 'attention' you may not specify any encoder-decoder arguments. This"
                             " argument is only for self-attention models.")
        if include_layers is None:
            include_layers = list(range(num_layers(attention)))
        attention = format_attention(attention, include_layers)
        attn_data.append(
                {
                    'name': None,
                    'attn': attention.tolist(),
                    'left_text': predict_sections_tokens,
                    'right_text': true_sections_tokens
                }
            )
    elif encoder_attention is not None or decoder_attention is not None or cross_attention is not None:
        if encoder_attention is not None:
            if encoder_tokens is None:
                raise ValueError("'encoder_tokens' required if 'encoder_attention' is not None")
            if include_layers is None:
                include_layers = list(range(num_layers(encoder_attention)))
            encoder_attention = format_attention(encoder_attention, include_layers)
            attn_data.append(
                {
                    'name': 'Encoder',
                    'attn': encoder_attention.tolist(),
                    'left_text': encoder_tokens,
                    'right_text': encoder_tokens
                }
            )
        if decoder_attention is not None:
            if decoder_tokens is None:
                raise ValueError("'decoder_tokens' required if 'decoder_attention' is not None")
            if include_layers is None:
                include_layers = list(range(num_layers(decoder_attention)))
            decoder_attention = format_attention(decoder_attention, include_layers)
            attn_data.append(
                {
                    'name': 'Decoder',
                    'attn': decoder_attention.tolist(),
                    'left_text': decoder_tokens,
                    'right_text': decoder_tokens
                }
            )
        if cross_attention is not None:
            if encoder_tokens is None:
                raise ValueError("'encoder_tokens' required if 'cross_attention' is not None")
            if decoder_tokens is None:
                raise ValueError("'decoder_tokens' required if 'cross_attention' is not None")
            if include_layers is None:
                include_layers = list(range(num_layers(cross_attention)))
            cross_attention = format_attention(cross_attention, include_layers)
            attn_data.append(
                {
                    'name': 'Cross',
                    'attn': cross_attention.tolist(),
                    'left_text': decoder_tokens,
                    'right_text': encoder_tokens
                }
            )
    else:
        raise ValueError("You must specify at least one attention argument.")

    if layer is not None and layer not in include_layers:
        raise ValueError(f"Layer {layer} is not in include_layers: {include_layers}")

    # Generate unique div id to enable multiple visualizations in one notebook
    vis_id = 'bertviz-%s'%(uuid.uuid4().hex)

    # Compose html
    if len(attn_data) > 1:
        options = '\n'.join(
            f'<option value="{i}">{attn_data[i]["name"]}</option>'
            for i, d in enumerate(attn_data)
        )
        select_html = f'Attention: <select id="filter">{options}</select>'
    else:
        select_html = ""
    vis_html = f"""      
        <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                Layer: <select id="layer"></select>
                {select_html}
            </span>
            <div id='vis'></div>
        </div>
    """

    for d in attn_data:
        attn_seq_len_left = len(d['attn'][0][0])
        if attn_seq_len_left != len(d['left_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_left} positions, while number of tokens is {len(d['left_text'])} "
                f"for tokens: {' '.join(d['left_text'])}"
            )
        attn_seq_len_right = len(d['attn'][0][0][0])
        if attn_seq_len_right != len(d['right_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_right} positions, while number of tokens is {len(d['right_text'])} "
                f"for tokens: {' '.join(d['right_text'])}"
            )
        if prettify_tokens:
            d['left_text'] = format_special_chars(d['left_text'])
            d['right_text'] = format_special_chars(d['right_text'])
    params = {
        'attention': attn_data,
        'default_filter': "0",
        'root_div_id': vis_id,
        'layer': layer,
        'heads': heads,
        'include_layers': include_layers
    }

    # require.js must be imported for Colab or JupyterLab:
    if html_action == 'view':
        display(HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
        display(HTML(vis_html))
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        display(Javascript(vis_js))

    elif html_action == 'return':
        html1 = HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>')

        html2 = HTML(vis_html)

        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        html3 = Javascript(vis_js)
        script = '\n<script type="text/javascript">\n' + html3.data + '\n</script>\n'

        head_html = HTML(html1.data + html2.data + script)
        return head_html

    else:
        raise ValueError("'html_action' parameter must be 'view' or 'return")
  
def read_id2station(dir = "data/trajFujianV2/stations2id.json"):
    with open(dir, 'r') as f:
        topo = json.load(f)
    stations = pd.read_csv("data/etc_map_topo.csv")
    stations = stations[stations['is_deleted'] == 0]
    filter_list = set()
    for _, row in stations.iterrows():
        if row['from_type'] == 3 and row['from_virtual'] == 0:
            filter_list.add(row['from_id'])
        if row['to_type'] == 3 and row['to_virtual'] == 0:
            filter_list.add(row['to_id'])
    enstationids = []
    for k, v in topo.items():
        if k.endswith('EN') or k in filter_list:
            enstationids.append(v)
    id2station = {v: k for k, v in topo.items()}
    return id2station, enstationids

def read_exstations(dir = "data/trajFujianV2/stations2id.json"):
    with open(dir, 'r') as f:
        topo = json.load(f)
    stations = pd.read_csv("data/etc_map_topo.csv")
    stations = stations[stations['is_deleted'] == 0]
    filter_list = set()
    for _, row in stations.iterrows():
        if row['from_type'] == 4 and row['from_virtual'] == 0:
            filter_list.add(row['from_id'])
        if row['to_type'] == 4 and row['to_virtual'] == 0:
            filter_list.add(row['to_id'])
    exstationids = set()
    for k, v in topo.items():
        if k.endswith('EX') or k in filter_list:
            exstationids.add(v)
    return exstationids
  
def draw_heatmap():
    random_seed(42)
    path = "./config/Transformer.ini"
    conf = read_config(path)
    tokenizer = Trajtokenizer(conf)
    model_path = os.path.join(conf['TRAIN']['save_path'], "best_model.pth")
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    mean = float(conf["DATASET"]["mean"])
    std = float(conf["DATASET"]["std"])
    load_data = eval(model_conf["MODEL"]["load_data_method"])
    train_loader, dev_loader, test_loader, history_loader, train_num = load_data(conf)
    model = eval(conf["TRAIN"]["model_name"])(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    select_index = 8
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src = batch['src'][select_index:select_index+1]
            enc_src = batch['enc_src'][select_index:select_index+1]
            tgt = batch['tgt'][select_index:select_index+1]
            enc_DoW = batch['enc_DoW'][select_index:select_index+1]
            enc_HoD = batch['enc_HoD'][select_index:select_index+1]
            travel_index = batch['travel_index'][select_index:select_index+1]
            # travel_mask = batch['travel_mask'][select_index:select_index+1]
            intervals = batch['intervals'][select_index:select_index+1]
            enc_intervals = batch['enc_intervals'][select_index:select_index+1]
            intervals_tgt = intervals.clone()
            # output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, travel_index, travel_mask, intervals, tgt)
            output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals, travel_index)
            intervals_output = exp_unnormalize(intervals_output, 0.0055730214824861605, mean, std)
            # intervals_output = unnormalize(intervals_output, mean, std)
            # intervals_output = MinMaxUnnormalize(intervals_output, mean, std)
            intervals_tgt = exp_unnormalize(intervals_tgt, 0.0055730214824861605, mean, std)
            # intervals_tgt = unnormalize(intervals_tgt, mean, std)
            # intervals_tgt = MinMaxUnnormalize(intervals_tgt, mean, std)
            break
    cache = get_local.cache
    index = torch.sum(src != 0, dim = -1)
    stationAtten = cache['StaionsAttensionLayer.forward']
    trajDecoderAtten = cache['trajDecoderLayer.forward']
    index = int(index)
    row_indices = torch.cat([torch.arange(0, index), torch.arange(256, 256+index)])
    col_indices = torch.cat([torch.arange(0, index), torch.arange(256, 256+index)])
    stationAtten = torch.tensor(stationAtten)[:, :, :, row_indices, :][:, :, :, :, col_indices]
    trajDecoderAtten = torch.tensor(trajDecoderAtten)[:,:,:,:index,:index]
    src_tokens = tokenizer.enuntokenize(src[0])[:index]
    tgt_tokens = tokenizer.enuntokenize(tgt[0])[:index]
    predict_tokens = tokenizer.enuntokenize(torch.argmax(output[0], dim = -1))[:index]
    intervals_tgt = intervals_tgt.squeeze()[:index]
    intervals_output = intervals_output.squeeze()[:index]
    true_sections = []
    for i in range(0, len(src_tokens)):
        true_sections.append(src_tokens[i] + '_' + tgt_tokens[i] + '_' + str(float(intervals_tgt[i])))
    
    predict_sections_tokens = []
    for i in range(0, len(src_tokens)):
        predict_sections_tokens.append(src_tokens[i] + '_' + predict_tokens[i] + '_' + str(float(intervals_output[i])))
    
    stationAtten_html_view = stationAttenView(stationAtten, predict_sections_tokens+predict_sections_tokens, true_sections+true_sections, html_action='return')
    trajDecoderAtten_html_view = head_view(trajDecoderAtten, src_tokens, html_action='return')
    with open("./stationAttenView.html", 'w') as file:
        file.write(stationAtten_html_view.data)
    with open("./trajDecoderAtten.html", 'w') as file:
        file.write(trajDecoderAtten_html_view.data)
    
def compute_metrics(travel_predictions, travel_labels, travel_intervals_predictions, travel_intervals_labels):
    travels = dict()
    
    for i in range(len(travel_intervals_labels)):
        length = len(travel_intervals_labels[i])
        for j in range(length):
            if j in travels:
                time_pred = travel_intervals_predictions[i][j]
                time_label = travel_intervals_labels[i][j]
                station_pred = travel_predictions[i][j]
                station_label = travel_labels[i][j]
                travels[j]["time_pred"].append(time_pred)
                travels[j]["time_label"].append(time_label)
                travels[j]["station_pred"].append(station_pred)
                travels[j]["station_label"].append(station_label)
            else:
                time_pred = travel_intervals_predictions[i][j]
                time_label = travel_intervals_labels[i][j]
                station_pred = travel_predictions[i][j]
                station_label = travel_labels[i][j]
                travels[j] = {"time_pred":[time_pred], "time_label":[time_label], "station_pred":[station_pred], "station_label":[station_label]}
    #compute results
    results = []
    counts = []
    indexs = []
    for k, v in travels.items():
        indexs.append(k)
        counts.append(len(v["time_label"]))
        time_pred = torch.tensor(v["time_pred"])
        time_label = torch.tensor(v["time_label"])
        station_pred = torch.tensor(v["station_pred"])
        station_label = torch.tensor(v["station_label"])
        res = time_station_Accuracy(time_pred.flatten(), time_label.flatten(), station_pred, station_label, threshold = 1.5)
        results.append(float(res))
        
    dataframe = pd.DataFrame({'time_station_Accuracy': results, 'counts': counts}, index=indexs)

    dataframe.to_csv("results.csv")
    
    return dataframe


def draw_lineplot(dir = 'data/predictions'):
    predictions_path = os.path.join(dir, "predictions.npy")
    label_path = os.path.join(dir, "labels.npy")
    predictions = np.load(predictions_path, allow_pickle=True)
    travel_predictions = predictions.item()['travel_predictions']
    travel_intervals_predictions = predictions.item()['travel_intervals_predictions']
    
    labels = np.load(label_path, allow_pickle=True)
    travel_labels = labels.item()['travel_labels']
    travel_intervals_labels = labels.item()['travel_intervals_labels']
    
    dataframe = compute_metrics(travel_predictions, travel_labels, travel_intervals_predictions, travel_intervals_labels)
    dataframe = dataframe[:-1]
    
    # x = dataframe.index
    # y1 = dataframe["counts"]
    # y2 = dataframe["time_station_Accuracy"]
    
    #设置画布大小
    plt.figure(figsize=(10, 6), dpi = 500)
    #画柱形图
    ax1 = plt.gca()
    # ax1.bar(x, y1,alpha=.7,color='g')
    sns.barplot(x=dataframe.index, y='counts', data=dataframe, color='#388db8', edgecolor=None, alpha = 0.7, width=1, label = 'History counts', ax=ax1)
    ax1.set_ylabel("History counts", color='#388db8')  # 设置左侧 Y 轴标签
    ax1.tick_params(axis='y', labelcolor='#388db8')  # 设置左侧 Y 轴刻度颜色
    #ax1.set_title("数据统计",fontsize='20')
    #画折线图 
    ax2 = ax1.twinx()   #组合图必须加这个
    # ax2.plot(x, y2, 'r',ms=10)
    sns.lineplot(x=dataframe.index, y='time_station_Accuracy', color='#cc529c', linewidth=2,  data=dataframe, label = 'TSA', ax=ax2, legend=False)
    ax2.set_ylabel("History Count", color='#cc529c')  # 设置右侧 Y 轴标签
    ax2.tick_params(axis='y', labelcolor='#cc529c')  # 设置右侧 Y 轴刻度颜色
    
    # 设置 X 轴的标签间隔（每隔 2 个显示一个标签）
    # plt.xticks(ticks=range(0, len(dataframe.index), 5), labels=dataframe.index[::5], rotation=45)
    custom_ticks = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    custom_labels = [
        r"$1$",  # 1
        r"$5$",     # 5
        r"$10$",     # 10
        r"$20$",     # 20
        r"$30$",     # 30
        r"$50$",     # 50
        r"$75$",     # 75
        r"$100$",     # 100
        r"$150$",     # 150
        r"$200$",     # 200
        r"$250$",     # 250
    ]
    plt.xticks(custom_ticks, labels=custom_labels)

    # 合并图例，并设置位置为左下角
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, 
        labels1 + labels2, 
        loc='center right',  # 设置图例位置为左下角
        bbox_to_anchor=(1, 0.3)  # 微调图例位置
    )
    save_path = os.path.join(dir, "lineplot.png")
    plt.savefig(save_path)


def read_data(path = "data/predictions/TrajPredTransformerV1", type = "TrajPredTransformer"):
    #EasyTemporalPointProcess 在预测的时候返回的结果少一第一个位置的数据
    if type == "EasyTemporalPointProcess":
        with open(path, 'rb') as file:
            try:
                data = pickle.load(file, encoding='latin-1')
            except Exception:
                data = pickle.load(file)
        new_travel_predictions = []
        new_travel_labels = []
        new_travel_intervals_predictions = []
        new_travel_intervals_labels = []
        # for i in range(len(data['pred'])):
        mask = (data['label'][1][:,30:254] != 0.0)
        #优于our的方法最大长度256，为了保证结果一致，取最后的225个数据, 30:254后面用254是因为起点是30，不是后面的31
        new_travel_predictions.extend(data['pred'][1][:,30:254][mask])
        new_travel_labels.extend(data['label'][1][:,30:254][mask])
        new_travel_intervals_predictions.extend(data['pred'][0][:,30:254][mask])
        new_travel_intervals_labels.extend(data['label'][0][:,30:254][mask])
    elif type == "based-transformer":
        new_travel_predictions, new_travel_labels, new_travel_intervals_predictions, new_travel_intervals_labels = read_res(path)
    else:
        travel_predictions, travel_labels, travel_intervals_predictions, travel_intervals_labels = read_res(path)
        new_travel_predictions = []
        new_travel_labels = []
        new_travel_intervals_predictions = []
        new_travel_intervals_labels = []
        count = 0
        for i in range(len(travel_intervals_labels)):
            length = len(travel_intervals_labels[i])
            if length >= 31:
                new_travel_predictions.extend(travel_predictions[i][31:255])
                new_travel_labels.extend(travel_labels[i][31:255])
                new_travel_intervals_predictions.extend(travel_intervals_predictions[i][31:255])
                new_travel_intervals_labels.extend(travel_intervals_labels[i][31:255])
            else:
                count += 1
        # print("count:", count)

    new_travel_predictions = torch.tensor(new_travel_predictions)
    new_travel_labels = torch.tensor(new_travel_labels)
    new_travel_intervals_predictions = torch.tensor(new_travel_intervals_predictions).squeeze(-1)
    new_travel_intervals_labels = torch.tensor(new_travel_intervals_labels).squeeze(-1)
    #filter data
    id2station, enstationids = read_id2station()
    enstationids = filter_data(new_travel_labels, enstationids)

    return new_travel_predictions, new_travel_labels, new_travel_intervals_predictions, new_travel_intervals_labels, enstationids
def draw_error_distribution():
    new_travel_predictions, new_travel_labels, new_travel_intervals_predictions, new_travel_intervals_labels, enstationids = read_data(path = "data/predictions/TrajPredTransformerV1", type = "TrajPredTransformer")
    # new_travel_predictions, new_travel_labels, new_travel_intervals_predictions, new_travel_intervals_labels, enstationids = read_data(path = "data/predictions/Transformer", type = "based-transformer")
    # en_travel_time = pd.DataFrame({'en_travel_time_error':list((new_travel_intervals_predictions - new_travel_intervals_labels)[enstationids].detach().numpy())})
    gantry_travel_time = pd.DataFrame({'gantry_travel_time_error':list(abs((new_travel_intervals_predictions - new_travel_intervals_labels))[~enstationids].detach().numpy())})
    # gantry_travel_time = pd.DataFrame({'new_travel_intervals_predictions':list(new_travel_intervals_predictions[~enstationids].detach().numpy()),
    #                                    'new_travel_intervals_labels':list(new_travel_intervals_labels[~enstationids].detach().numpy()),})
    en_travel_time = pd.DataFrame({'new_travel_intervals_predictions':list(new_travel_intervals_predictions[enstationids].detach().numpy()),
                                       'new_travel_intervals_labels':list(new_travel_intervals_labels[enstationids].detach().numpy()),})

    plt.figure(figsize=(5, 5), dpi=500)
    
    # sns.kdeplot(x = gantry_travel_time['new_travel_intervals_predictions'], y = gantry_travel_time['new_travel_intervals_labels'], fill=True, 
    #             color="#2e6eba", bw_adjust=1.5, gridsize=100, cbar=True, label="gantry")
    # sns.jointplot(data = gantry_travel_time[:1000], x = 'new_travel_intervals_predictions', y = 'new_travel_intervals_labels', kind = 'kde', color = "#2e6eba", label = "gantry")
    sns.displot(gantry_travel_time, kde = True, bins=850)
    #添加y=x的参考线
    # plt.plot([0, 30], [0, 30], color='black', linewidth=1)
    # sns.kdeplot(x = en_travel_time['new_travel_intervals_predictions'], y = en_travel_time['new_travel_intervals_labels'], fill=True, color="#7f5cc3", label="en")
    # sns.kdeplot(en_travel_time, shade=True, color="#7f5cc3", alpha=0.3, label="en")
    #设置x轴范围
    plt.xlim(0, 30)
    # plt.ylim(0, 5)
    # plt.xscale("log")
    # custom_ticks = [1, 2, 5, 10, 20, 50, 100, 400, 10**3, 10**4]
    # custom_labels = [
    #     r"$1$",  # 1
    #     r"$2$",     # 2
    #     r"$5$",     # 5
    #     r"$10$",  # 10
    #     r"$20$",  # 20
    #     r"$50$",  # 50
    #     r"$10^2$",  # 100
    #     r"$400$",  # 400
    #     r"$10^3$",  # 10^3
    #     r"$10^4$",  # 10^4
    # ]
    # plt.xticks(custom_ticks, labels=custom_labels)
    plt.title("")
    plt.xlabel("prediction")
    plt.ylabel("true")
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/predictions/TrajPredTransformerV1/travel_time_error_kde.png')

def draw_traj():
    # 经纬度转换为弧度
    def deg_to_rad(deg):
        return deg * (math.pi / 180)

    # 哈夫辛公式计算两点间的距离
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # 地球半径（公里）
        
        # 将经纬度转换为弧度
        lat1 = deg_to_rad(lat1)
        lon1 = deg_to_rad(lon1)
        lat2 = deg_to_rad(lat2)
        lon2 = deg_to_rad(lon2)
        
        # 计算纬度和经度的差值
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # 哈夫辛公式
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # 计算距离
        distance = R * c
        return distance
    
    _topo = pd.read_csv("data/etc_map_topo.csv")
    _topo = _topo[_topo['is_deleted'] == 0]
    _topo = _topo[(_topo['from_virtual'] == 0) & (_topo['to_virtual'] == 0)]
    topo = dict()
    interval_time = 60 * 3

    enstation = 495
    id2station, enstationids = read_id2station()
    
    lat = 25.9274
    lng = 119.3457
    max_distance = interval_time/60 * 120
    for _, row in _topo.iterrows():
        from_distance = haversine(lat,lng,row['from_lat'],row['from_lng'])
        to_distance = haversine(lat,lng,row['to_lat'],row['to_lng'])
        if from_distance < max_distance and to_distance < max_distance:
            topo[(row['from_id'],row['to_id'])] = eval(row['path'])
        
    travel_predictions, travel_labels, travel_intervals_predictions, travel_intervals_labels, enstationids = read_data()
    exstations = read_exstations()
    
    # user_ids = []
    # destination = []
    total_p_trips = []
    total_t_trips = []
    travel_p_costs = []
    travel_t_costs = []
    

    flag = False
    _total_p_trips = []
    _total_t_trips = []
    _travel_p_costs = []
    _travel_t_costs = []
    for i, station in tqdm(enumerate(travel_labels)):
        if flag == True:
            cost += travel_intervals_labels[i]
            if cost >= interval_time:
            # if cost >= interval_time or station in exstations:
                flag = False
                total_p_trips.append(_total_p_trips)
                total_t_trips.append(_total_t_trips)
                travel_p_costs.append(_travel_p_costs)
                travel_t_costs.append(_travel_t_costs)
                _total_p_trips = []
                _total_t_trips = []
                _travel_p_costs = []
                _travel_t_costs = []
            else:
                _total_p_trips.append(id2station[int(station)])
                _total_t_trips.append(id2station[int(travel_predictions[i])])
                _travel_p_costs.append(float(travel_intervals_labels[i]))
                _travel_p_costs.append(float(travel_intervals_predictions[i]))
            
        if station == enstation:
            cost = 0
            _total_p_trips.append(id2station[int(station)])
            _total_t_trips.append(id2station[int(travel_predictions[i])])
            _travel_p_costs.append(0.0)
            _travel_t_costs.append(0.0)
            flag = True

    trajectories = dict()
    for i in range(len(total_t_trips)):
        for node in zip(total_t_trips[i][:-1], total_t_trips[i][1:]):
            if node in topo:
                if node in trajectories:
                    trajectories[node] += 1
                else:
                    trajectories[node] = 1
    # 创建一个图形
    plt.figure(figsize=(10, 8))
    # 遍历每一条轨迹
    max_count = max(trajectories.values())
    for trajectory, count in trajectories.items():
        # 提取纬度和经度
        latitude = [point[0] for point in topo[trajectory]]
        longitude = [point[1] for point in topo[trajectory]]
        
        # 绘制轨迹，使用不同颜色或者样式
        plt.plot(longitude, latitude, color='black', marker='o', linestyle='-', markersize=4*count/max_count, alpha=0.1)

    # 去掉所有外框
    ax = plt.gca()  # 获取当前坐标轴对象
    ax.spines['top'].set_visible(False)  # 去掉顶部边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['bottom'].set_visible(False)  # 去掉底部边框

    # 去掉刻度线
    ax.tick_params(axis='both', which='both', length=0)  # 去掉刻度线

    # 去掉坐标轴标签
    ax.set_xticklabels([])  # 去掉x轴标签
    ax.set_yticklabels([])  # 去掉y轴标签
    # plt.xlim(116,120)
    # plt.ylim(24,28)
    # 设置图形的标题和坐标轴标签
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.savefig('data/predictions/TrajPredTransformerV1/truetraj.png')
    
    trajectories = dict()
    for i in range(len(total_p_trips)):
        for node in zip(total_p_trips[i][:-1], total_p_trips[i][1:]):
            if node in topo:
                if node in trajectories:
                    trajectories[node] += 1
                else:
                    trajectories[node] = 1
    # 创建一个图形
    plt.figure(figsize=(10, 8))
    # 遍历每一条轨迹
    for trajectory, count in trajectories.items():
        # 提取纬度和经度
        latitude = [point[0] for point in topo[trajectory]]
        longitude = [point[1] for point in topo[trajectory]]
        
        # 绘制轨迹，使用不同颜色或者样式
        plt.plot(longitude, latitude, color='black', marker='o', linestyle='-', markersize=4*count/max_count, alpha=0.1)
    
    # 去掉所有外框
    ax = plt.gca()  # 获取当前坐标轴对象
    ax.spines['top'].set_visible(False)  # 去掉顶部边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['bottom'].set_visible(False)  # 去掉底部边框

    # 去掉刻度线
    ax.tick_params(axis='both', which='both', length=0)  # 去掉刻度线

    # 去掉坐标轴标签
    ax.set_xticklabels([])  # 去掉x轴标签
    ax.set_yticklabels([])  # 去掉y轴标签
    # 设置图形的标题和坐标轴标签
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.savefig('data/predictions/TrajPredTransformerV1/predtraj.png')
    

if __name__ == "__main__":
    # draw_heatmap()
    # draw_lineplot("data/predictions/TrajPredTransformerV1")
    # draw_error_distribution()
    draw_traj()
