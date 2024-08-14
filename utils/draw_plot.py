import os,sys,torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tokenizer import Entokenizer
from visualizer import get_local
get_local.activate()
from train import TrajPredTransformerV1, random_seed, load_ETCEn_dataV1, read_config
from utils.metrics import unnormalize
from bertviz import head_view, model_view
from bertviz.head_view import *
from bertviz.neuron_view import show

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
    
def draw_heatmap():
    random_seed(42)
    path = "./config/config.ini"
    conf = read_config(path)
    tokenizer = Entokenizer(conf)
    train_loader, dev_loader, test_loader, history_loader, train_num = load_ETCEn_dataV1(conf)
    model_path = "weights/new1/best_model.pth"
    model_conf = read_config(conf["TRAIN"]["model_config_path"])
    model = TrajPredTransformerV1(model_conf["MODEL"])
    model.to(conf["DATASET"]["device"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    select_index = 6
    with torch.no_grad():
        for i, batch in enumerate(history_loader):
            src = batch['src'][select_index:select_index+1]
            enc_src = batch['enc_src'][select_index:select_index+1]
            tgt = batch['tgt'][select_index:select_index+1]
            enc_DoW = batch['enc_DoW'][select_index:select_index+1]
            enc_HoD = batch['enc_HoD'][select_index:select_index+1]
            intervals = batch['intervals'][select_index:select_index+1]
            enc_intervals = batch['enc_intervals'][select_index:select_index+1]
            intervals_tgt = batch['intervals'][select_index:select_index+1]
            output, intervals_output =model(enc_src, enc_intervals, src, enc_HoD, enc_DoW, intervals)
            intervals_output = unnormalize(intervals_output)
            intervals_tgt = unnormalize(intervals_tgt)
            break
    cache = get_local.cache
    index = torch.sum(src != 0, dim = -1)
    stationAtten = cache['StaionsAttensionLayer.forward']
    trajDecoderAtten = cache['trajDecoderLayer.forward']
    stationAtten = torch.tensor(stationAtten)[:,:,:,:index,:index]
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
    
    stationAtten_html_view = stationAttenView(stationAtten, predict_sections_tokens, true_sections, html_action='return')
    trajDecoderAtten_html_view = head_view(trajDecoderAtten, true_sections, html_action='return')
    with open("./stationAttenView.html", 'w') as file:
        file.write(stationAtten_html_view.data)
    with open("./trajDecoderAtten.html", 'w') as file:
        file.write(trajDecoderAtten_html_view.data)
    
if __name__ == "__main__":
    draw_heatmap()
