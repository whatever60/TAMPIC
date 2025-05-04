#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycirclize import Circos
from pycirclize.utils import load_example_tree_file

tree_file = load_example_tree_file("alphabet.nwk")
circos, tv = Circos.initialize_from_tree(
    tree_file,
    start=20,                                # Default: 0
    end=340,                                 # Default: 360
    r_lim=(10, 100),                         # Default: (50, 100)
    line_kws=dict(color="red", lw=2),        # Default: {}, Change color & linewidth
    align_line_kws=dict(ls="dashdot", lw=1), # Default: {}, Change linestyle & linewidth
)
fig = circos.plotfig()


# In[2]:


from pycirclize import Circos
from pycirclize.utils import load_example_tree_file
import matplotlib.pyplot as plt

# Create 2x2 polar subplots
fig = plt.figure(figsize=(16, 16))
fig.subplots_adjust(wspace=0.05, hspace=0.05)
ax_list = fig.subplots(2, 2, subplot_kw=dict(polar=True)).flatten()

# Define 4 types kwargs for `Circos.initialize_from_tree()` method
kwargs_list = [
    dict(outer=True, align_leaf_label=True, ignore_branch_length=False),
    dict(outer=True, align_leaf_label=False, ignore_branch_length=False),
    dict(outer=False, align_leaf_label=True, ignore_branch_length=False),
    dict(outer=True, align_leaf_label=True, ignore_branch_length=True),
]

# Plot trees with different kwargs
tree_file = load_example_tree_file("alphabet.nwk")
for ax, kwargs in zip(ax_list, kwargs_list):
    circos, tv = Circos.initialize_from_tree(tree_file, r_lim=(60, 100), **kwargs)
    kwargs_text = "\n".join([f"{k}: {v}" for k, v in kwargs.items()])
    circos.text(kwargs_text, size=14)
    circos.plotfig(ax=ax)


# In[3]:


from pycirclize import Circos
from pycirclize.utils import load_example_tree_file
import matplotlib.pyplot as plt

# Create 2x2 polar subplots
fig = plt.figure(figsize=(16, 16))
fig.subplots_adjust(wspace=0.05, hspace=0.05)
ax_list = fig.subplots(2, 2, subplot_kw=dict(polar=True)).flatten()

# Define 4 types kwargs for `Circos.initialize_from_tree()` method
kwargs_list = [
    dict(reverse=False, ladderize=False),
    dict(reverse=True, ladderize=False),
    dict(reverse=False, ladderize=True),
    dict(reverse=True, ladderize=True),
]

# Plot trees with different kwargs
tree_file = load_example_tree_file("alphabet.nwk")
for ax, kwargs in zip(ax_list, kwargs_list):
    circos, tv = Circos.initialize_from_tree(tree_file, line_kws=dict(lw=1), **kwargs)
    kwargs_text = "\n".join([f"{k}: {v}" for k, v in kwargs.items()])
    circos.text(kwargs_text, size=14)
    circos.plotfig(ax=ax)


# In[4]:


import pandas as pd
import numpy as np
from pycirclize import Circos
from pycirclize.utils import load_example_tree_file, ColorCycler
from matplotlib.lines import Line2D

np.random.seed(0)

tree_file = load_example_tree_file("alphabet.nwk")
circos, tv = Circos.initialize_from_tree(
    tree_file,
    start=5,
    end=355,
    r_lim=(5, 60),
    line_kws=dict(lw=1),
)

# Change label color
tv.set_node_label_props("A", color="red")
# Change label color & size
tv.set_node_label_props("D", color="blue", size=25)
# Hide label
tv.set_node_label_props("G", size=0)
# Change label colors
ColorCycler.set_cmap("tab10")
for name in list("MNOPQRSTUVWXY"):
    tv.set_node_label_props(name, color=ColorCycler())

# Change line color on [A,B,C,D,E,F] MRCA(Most Recent Common Ancestor) node and its descendent nodes
tv.set_node_line_props(["A", "B", "C", "D", "E", "F"], color="red")
# Change line color & width on [G,I] MRCA node and its descendent nodes
tv.set_node_line_props(["G", "I"], color="blue", lw=2.0)
# Change line color & label color on [M,W] MRCA node and its descendent nodes
tv.set_node_line_props(["M", "W"], color="green", apply_label_color=True)
# Change line color & label color on [R,T] MRCA node and its descendent nodes
tv.set_node_line_props(["R", "T"], color="purple", apply_label_color=True)
# Change line color & style [X,Y] MRCA node
tv.set_node_line_props(["X", "Y"], color="orange", descendent=False, ls="dotted")

# Plot markers on [A,B,C,D,E,F] MRCA(Most Recent Common Ancestor) node and its descendent nodes
tv.marker(["A", "B", "C", "D", "E", "F"], color="salmon")
# Plot square marker on [G,K] MRCA node
tv.marker(["G", "K"], color="orange", marker="s", size=8, descendent=False)
# Plot star markers on [X,Y] MRCA node and its descendent nodes
tv.marker(["X", "Y"], color="lime", marker="*", size=10, ec="black", lw=0.5)
# Plot colored markers on M,N,O,P,Q,R,S,T,U,V,W leaf nodes
ColorCycler.set_cmap("Set3")
for leaf_name in list("MNOPQRSTUVW"):
    tv.marker(leaf_name, color=ColorCycler(), ec="black", lw=0.5)

# Plot highlight on [A,B,C,D,E,F] MRCA(Most Recent Common Ancestor) node
tv.highlight(["A", "B", "C", "D", "E", "F"], color="salmon")
# Plot highlight on [G,K] MRCA node
tv.highlight(["G", "K"], color="orange")
# Plot highlight on L node with '//' hatch pattern
tv.highlight("L", color="lime", hatch="//", ec="white")
# Plot highlight on [N,W] MRCA node with edge line
tv.highlight(["N", "W"], color="lightgrey", alpha=0.5, ec="red", lw=0.5)

# Create example dataframe for heatmap & bar plot
df = pd.DataFrame(
    dict(
        s1=np.random.randint(0, 100, tv.leaf_num),
        s2=np.random.randint(0, 100, tv.leaf_num),
        s3=np.random.randint(0, 100, tv.leaf_num),
        count=np.random.randint(1, 10, tv.leaf_num),
    ),
    index=tv.leaf_labels,
)
print(df.head())

# Plot bar (from `count` column data)
sector = tv.track.parent_sector
bar_track = sector.add_track((85, 100), r_pad_ratio=0.1)
bar_track.axis()
bar_track.grid()
x = np.arange(0, tv.leaf_num) + 0.5
y = df["count"].to_numpy()
bar_track.bar(x, y, width=0.3, color="orange")

# Plot heatmaps (from `s1, s2, s3` column data)
track1 = sector.add_track((80, 85))
track1.heatmap(df["s1"].to_numpy(), cmap="Reds", show_value=True, rect_kws=dict(ec="grey", lw=0.5))
track2 = sector.add_track((75, 80))
track2.heatmap(df["s2"].to_numpy(), cmap="Blues", show_value=True, rect_kws=dict(ec="grey", lw=0.5))
track3 = sector.add_track((70, 75))
track3.heatmap(df["s3"].to_numpy(), cmap="Greens", show_value=True, rect_kws=dict(ec="grey", lw=0.5))

# Plot track labels
circos.text("count", r=bar_track.r_center, color="orange")
circos.text("s1", r=track1.r_center, color="red")
circos.text("s2", r=track2.r_center, color="blue")
circos.text("s3", r=track3.r_center, color="green")

vmin, vmax, cmap = 0, 100, "bwr"
circos.colorbar((0.4, 0.495, 0.2, 0.01), vmin=vmin, vmax=vmax, cmap=cmap, orientation="horizontal")

fig = circos.plotfig()

group_name2species_list = dict(
    Monotremata=["Tachyglossus_aculeatus", "Ornithorhynchus_anatinus"],
    Marsupialia=["Monodelphis_domestica", "Vombatus_ursinus"],
    Xenarthra=["Choloepus_didactylus", "Dasypus_novemcinctus"],
    Afrotheria=["Trichechus_manatus", "Chrysochloris_asiatica"],
    Euarchontes=["Galeopterus_variegatus", "Theropithecus_gelada"],
    Glires=["Oryctolagus_cuniculus", "Microtus_oregoni"],
    Laurasiatheria=["Talpa_occidentalis", "Mirounga_leonina"],
)

# Set tree line color & label color
ColorCycler.set_cmap("tab10")
group_name2color = {name: ColorCycler() for name in group_name2species_list.keys()}
for group_name, species_list in group_name2species_list.items():
    color = group_name2color[group_name]
_ = circos.ax.legend(
    handles=[Line2D([], [], label=n, color=c) for n, c in group_name2color.items()],
    labelcolor=group_name2color.values(),
    fontsize=6,
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
)


# In[5]:


# read amplicon_info

import glob
import json

import numpy as np
import pandas as pd
import dendropy
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data_dir = "/home/ubuntu/data/camii"
amplicon_type = "16s"
isolates_info = []
taxa_info = []
levels = ["domain", "phylum", "class", "order", "family", "genus", "species"]
for amplicon_info_path in glob.glob(f"{data_dir}/*/metadata/amplicon_info.json"):
    project = amplicon_info_path.split("/")[-3]
    with open(amplicon_info_path) as f:
        amplicon_info = json.load(f)
    for plate, plate_data in amplicon_info.items():
        for isolate, isolate_data in plate_data.items():
            isolate_info = {
                "project": project,
                "plate": plate,
                "isolate": isolate,
            }
            for taxonomy in isolate_data[amplicon_type]["taxonomies"]:
                taxa_info.append(taxonomy["taxon"])


# In[7]:


train_df_path = "/home/ubuntu/dev/TAMPIC/tb_logs/20240911_TAMPIC_16s_genus_ablation_crop-size_512_no-hsi/version_1/predictions/_train_all.csv"
val_easy_info_paths = glob.glob("/home/ubuntu/dev/TAMPIC/tb_logs/20240911_TAMPIC_16s_genus_ablation_crop-size_512_no-hsi/version_1/predictions/val_easy.csv")
train_df = pd.read_csv(train_df_path)
train_df["label_clean"].value_counts()


# In[8]:


taxa_df = pd.DataFrame(taxa_info, columns=levels).drop("species", axis=1).drop_duplicates()
# drop the row if all values are "unknown"
taxa_df = taxa_df.replace("unknown", np.nan).dropna(how="all")
taxa_df = taxa_df.sort_values(by=taxa_df.columns.tolist())


# In[9]:


def df2tree(df: pd.DataFrame) -> dendropy.Tree:
    taxa = []
    for level in df.columns:
        taxa.extend(df[level].dropna().unique())
    taxon_namespace = dendropy.TaxonNamespace(taxa)
    tree = dendropy.Tree(taxon_namespace=taxon_namespace)
    taxon_added = {l: set() for l in df.columns}
    for idx, row in df.iterrows():
        edge_length = 1
        node_current = tree.seed_node
        for level, taxon in row.items():
            if pd.notna(taxon):
                if taxon not in taxon_added[level]:
                    taxon_added[level].add(taxon)
                    node = dendropy.Node(edge_length=edge_length)
                    node.taxon = taxon_namespace.get_taxon(taxon)
                    node_current.add_child(node)
                    node_current = node
                else:
                    node_current = tree.find_node_for_taxon(taxon_namespace.get_taxon(taxon))
                edge_length = 1
            else:
                edge_length += 1
    return tree


# In[10]:


tree = df2tree(taxa_df)
print(tree.as_ascii_plot())


# In[11]:


taxa_df = taxa_df.query("genus.isin(@train_df['label_clean'].unique())")
tree = df2tree(taxa_df)
print(tree.as_ascii_plot())
tree.write(
    path="./tb_logs/20240911_TAMPIC_16s_genus_ablation_crop-size_512_no-hsi/version_1/predictions/label_clean.nwk",
    schema="newick",
)


# In[12]:


val_easy_info = pd.read_csv(
    "./tb_logs/20240911_TAMPIC_16s_genus_ablation_crop-size_512_no-hsi/version_1/predictions/val-easy_epoch-299_info.csv"
)
val_easy_logits = pd.read_parquet(
    "./tb_logs/20240911_TAMPIC_16s_genus_ablation_crop-size_512_no-hsi/version_1/predictions/val-easy_epoch-299_logits.parquet"
)
labels = val_easy_logits.columns[val_easy_info["label"].to_numpy()].to_numpy()
preds = val_easy_logits.idxmax(axis=1).to_numpy()
base_level = "genus"
# ground truth is in the label column of val_easy_info, and the predicted logits are in val_easy_logits
# the prediction is at genus level, but we can also evaluate accuracy at higher levels by aggregating the predictions
# and treating prediction as correct if prediction has the same label as the ground truth at the higher level
level2taxon2odd = {}
for level in taxa_df:
    # aggregate logit of the same label at this level
    _taxa_df = taxa_df.copy()
    _taxa_df.index = _taxa_df[base_level]
    label2taxon = _taxa_df[level].to_dict()
    label2taxon["others"] = "others"
    _labels = np.array([label2taxon[label] for label in labels])
    _preds = np.array([label2taxon[label] for label in preds])
    null_acc = 1 / np.unique(_labels).shape[0]
    print(level, null_acc)
    accs = (
        pd.DataFrame({"label": _labels, "pred": _preds})
        .groupby("label")
        .apply(lambda x: (x["label"] == x["pred"]).mean())
    )
    # odds = accs / null_acc
    odds = accs
    level2taxon2odd[level] = odds.to_dict()
level2taxon2odd


# In[64]:


circos, tv = Circos.initialize_from_tree(
    "./tb_logs/20240911_TAMPIC_16s_genus_ablation_crop-size_512_no-hsi/version_1/predictions/label_clean.nwk",
    start=15,
    end=360,
    r_lim=(5, 60),
    line_kws=dict(lw=1),
    leaf_label_rmargin=5,
)
# add dots to the leaves:
# - size proportional to the odd ratio of model prediction, i.e., accuracy over random (also add this to each internal node)
# - color with a color map like dark showing a taxonomy level like class (use an argument for that, skip if None)
# color node line accordingly
# highlight with another color map like tab showing plylum, with some opacity
# add legend for size and color
# add a bar plot showing number of samples, i.e., train_df["label_clean"].value_counts()

# let's figure out a way to do this right with tv, dendrogram, taxa_df and train_df
# first let's leaf color and line color
leaf_level = "genus"
level_to_color_node = "phylum"
level_to_color_line = "class"
level_to_highlight = "phylum"
color_map_node = "tab10"
color_map_line = "Dark2"
color_map_highlight = "Set3"

# ColorCycler.set_cmap(color_map_highlight)
# for taxon, taxon_level in taxa_df.query(
#     "genus.isin(@train_df['label_clean'].unique())"
# ).groupby(level_to_highlight):
#     tv.highlight(
#         taxon,
#         # color=ColorCycler(
#         #     color_map_highlight, len(taxa_level[level_to_highlight].unique())
#         # ),
#         color=ColorCycler(),
#         alpha=0.5,
#         ec="grey",
#         lw=0.5,
#     )

# ColorCycler.set_cmap(color_map_line)
# for taxon, taxon_level in taxa_df.query(
#     "genus.isin(@train_df['label_clean'].unique())"
# ).groupby(level_to_color_line):
#     tv.set_node_line_props(
#         taxon,
#         color=ColorCycler(),
#     )


ColorCycler.set_cmap(color_map_node)
taxon2color = {}
for taxon, taxon_level in taxa_df.groupby(level_to_color_node):
    taxa = taxa_df.query(f"{level_to_color_node} == @taxon")
    # only take the taxon column and following columns
    col_idx = taxa.columns.get_loc(level_to_color_node)
    color = ColorCycler()
    for taxon in taxa.iloc[:, col_idx:].to_numpy().flatten():
        taxon2color[taxon] = color

odds = []
for level in taxa_df.columns[1:]:  # skip domain, start from phylum
    odds.extend(
        [
            level2taxon2odd[level][taxon]
            for taxon in taxa_df[level].unique()
            if taxon in tv.all_node_labels
        ]
    )

odd2size = lambda odd: odd * 10 + 2

for level in taxa_df.columns[1:]:  # skip domain, start from phylum
    for taxon in taxa_df[level].unique():
        if taxon in tv.all_node_labels and taxon in taxon2color:
            tv.marker(
                taxon,
                size=odd2size(level2taxon2odd[level][taxon]),
                color=taxon2color[taxon],
                descendent=False,
            )

for leaf in tv.leaf_labels:
    tv.set_node_label_props(leaf, alpha=0.5)

# add bar plot showing number of samples
sector = tv.track.parent_sector
bar_track = sector.add_track((65, 100), r_pad_ratio=0.1)
bar_track.axis()
bar_track.grid(y_grid_num=3)
x = np.arange(0, tv.leaf_num) + 0.5
y = np.log10(train_df["label_clean"].value_counts()[tv.leaf_labels].to_numpy()) - 1
bar_track.bar(x, y, width=0.3, color="orange")
bar_track.yticks(
    np.array(list(range(1, 4))) - 1, labels=[f"$10^{i}$" for i in range(1, 4)]
)

circos.text(
    "Number of\ntraining samples",
    r=60,
    deg=10,
    adjust_rotation=True,
    orientation="vertical",
    # ha="center",
    # va="center",
    ma="center",
)

fig = circos.plotfig()

# add an axis for node color with dot
# node_color_legend_ax = fig.add_subplot([0.1, 0.1, 0.1, 0.1])
# node_color_legend_ax.axis("off")
# node_color_legend_ax.legend(
#     handles=[
#         Line2D([], [], marker="o", color=c, label=n, markersize=10)
#         for n, c in taxon2color.items()
#         if n in taxa_df[level_to_color_node].unique()
#     ],
#     fontsize=6,
#     loc="center",
#     bbox_to_anchor=(0.5, 0.5),
# )
ColorCycler.set_cmap(color_map_node)
_ = circos.ax.legend(
    handles=[
        Line2D([], [], label=n, color=c, marker="o", markersize=5, linestyle="None")
        for n, c in taxon2color.items()
        if n in taxa_df[level_to_color_node].unique()
    ],
    fontsize=10,
    loc="center",
    bbox_to_anchor=(1.1, 0.2),
    handleheight=2.0,
)
# add another legend for the size of the dots for 0.2, 0.4, 0.6, 0.8, 1.0
node_size_legend_ax = fig.add_axes(
    [
        0.85,
        0.7,
        0.1,
        0.1,
    ]
)
node_size_legend_ax.axis("off")
node_size_legend_ax.legend(
    handles=[
        Line2D(
            [],
            [],
            marker="o",
            color="black",
            label=f"{odd:.2f}",
            markersize=odd2size(odd),
            linestyle="None",
        )
        for odd in [0.2, 0.4, 0.6, 0.8, 1.0]
    ],
    fontsize=10,
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    handleheight=2.0,
)


# In[51]:


y


# In[52]:


taxa.iloc[:, col_idx:]


# In[41]:


sns.histplot(odds, bins=20)


# In[ ]:


tv.


# In[53]:


tv.all_node_labels


# In[ ]:




