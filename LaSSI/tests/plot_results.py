import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme, scale_y_log10, scale_color_brewer, \
    element_rect, scale_x_continuous, geom_point, element_text, scale_linetype_manual, element_blank, geom_rect


def main():
    font = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=8)
    bold_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=8)
    title_font = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=12)

    pd.set_option('display.max_columns', None)

    data = pd.read_csv('benchmarks/mar18-benchmark-added-logical.csv')  # FYI: mar18 is used in MDPI25 paper
    data = data.sort_values(by='Dataset')
    data = data.replace(0, np.nan)  # For generating/loading meuDB where values are 0
    averaged_data = data.groupby('Dataset', as_index=False).mean(numeric_only=True)

    # Create a new DataFrame for the GPT-3 data
    gpt3_data = pd.DataFrame({
        'Dataset': averaged_data['Dataset'].unique(),
        # 'GPT-3 training time': [34 * 24 * 60 * 60] * len(averaged_data['Dataset'].unique()) # 34 days to seconds
        'all-MiniLM-L6-v2/all-MiniLM-L12-v2/ training time': [10240000] * len(averaged_data['Dataset'].unique()),
        'all-roberta-large-v1 training time': [8192000] * len(averaged_data['Dataset'].unique())
        # TODO: Cannot get other training times
    })

    # Merge the GPT-3 data with the averaged data
    averaged_data = pd.merge(averaged_data, gpt3_data, on='Dataset', how='left')

    melted_data = averaged_data.melt(id_vars=['Dataset'], var_name='Phase', value_name='Time')
    melted_data['Dataset'] = pd.to_numeric(melted_data['Dataset'])  # Essential for highlighting last dataset on graph
    # melted_data['Dataset'] = pd.Categorical(melted_data['Dataset'], categories=sorted(data['Dataset'].unique()),
    #                                         ordered=True)
    melted_data['Phase'] = pd.Categorical(melted_data['Phase'], categories=list(averaged_data.columns[1:]), ordered=True)

    original_labels = list(averaged_data.columns[1:])
    line_types = ['solid'] * len(original_labels)
    for i, label in enumerate(original_labels):
        if 'training time' in label.lower():
            line_types[i] = 'dashed'
    line_type_dict = dict(zip(original_labels, line_types))

    # Highlight last dataset
    last_dataset_label = sorted(data['Dataset'].unique())[-1]
    last_dataset_data = melted_data[pd.to_numeric(melted_data['Dataset']) == last_dataset_label]
    last_dataset_data = last_dataset_data[last_dataset_data['Phase'].str.contains("training time") == False]
    y_min = last_dataset_data['Time'].min()
    y_max = last_dataset_data['Time'].max()

    last_dataset_data_meu = last_dataset_data[(last_dataset_data['Phase'] != "Loading meuDB")]
    total_time_last_dataset = last_dataset_data_meu['Time'].sum()
    print(f"Total time for the {last_dataset_label} dataset w/ MEU: {total_time_last_dataset/60} minutes")

    last_dataset_data_no_meu = last_dataset_data[(last_dataset_data['Phase'] != "Generating meuDB")]
    total_time_last_dataset = last_dataset_data_no_meu['Time'].sum()
    print(f"Total time for the {last_dataset_label} dataset w/out MEU: {total_time_last_dataset} minutes")

    plot = (
            ggplot(melted_data, aes(x='Dataset', y='Time', color='Phase', group='Phase')) +
            scale_x_continuous(breaks=sorted(melted_data['Dataset'].unique()), labels=sorted(data['Dataset'].unique()),
                               limits=(0, 205)) +
            geom_rect(aes(xmin=last_dataset_label - 5, xmax=last_dataset_label + 5, ymin=y_min / 3, ymax=y_max * 3),
                      color='red', fill='none', size=0.75) +
            geom_line(aes(linetype='Phase'), size=0.75) +
            geom_point(aes(shape="Phase"), size=1.75) +
            scale_y_log10(minor_breaks=[],
                          breaks=[10 ** x for x in range(-5, 7)],
                          labels=lambda l: ["{:.0e}".format(v).replace("+0", "+").replace("-0", "-") for v in l]) +
            scale_color_brewer(type='qual', palette='Dark2') +
            scale_linetype_manual(values=line_type_dict) +
            labs(title='LaSSI Phase Execution Times vs. Number of Sentences',
                 x='Number of sentences',
                 y='Time (seconds, log scale)',
                 color='Phase') +
            theme_minimal() +
            theme(
                plot_background=element_rect(fill='white', color="white"),
                text=element_rect(fontproperties=font),
                legend_text=element_text(ha='left'),
                axis_title_x=element_text(fontproperties=bold_font),
                axis_title_y=element_text(fontproperties=bold_font),
                legend_title=element_text(ha='left', fontproperties=bold_font),
                plot_title=element_text(ha='center', fontproperties=title_font),
                panel_border=element_blank(),
                legend_position='bottom',
                legend_direction='horizontal',
            )
            + guides(color=guide_legend(nrow=3), shape=guide_legend(nrow=3))
    )
    plot.save('performance_metrics_plot.png', dpi=1200, width=7.5, height=5)

if __name__ == "__main__":
    main()