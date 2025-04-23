import matplotlib.font_manager as fm
import pandas as pd
from plotnine import ggplot, aes, scale_y_log10, labs, geom_point, geom_line, \
    scale_x_continuous, element_text, element_blank, theme_minimal, theme, scale_color_brewer, element_rect, guides, \
    guide_legend

if __name__ == '__main__':
    font_regular = fm.FontProperties(fname='./fonts/Satoshi-Medium.ttf', size=8)
    font_bold = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=8)
    font_title = fm.FontProperties(fname='./fonts/Satoshi-Bold.ttf', size=12)

    df = pd.read_csv('./benchmarks/each_sentence/benchmark_results_200.csv')
    df_to_append = pd.read_csv(
        './benchmarks/each_sentence/java_graph_generation_benchmark_nvertices_vs_milliseconds.csv', names=['Vertices', 'Generating StanfordNLP representation'])

    df_to_append['Generating StanfordNLP representation'] = df_to_append['Generating StanfordNLP representation'] * 1000
    df = pd.concat([df, df_to_append], axis=1, ignore_index=False, sort=False)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df_mean = df.groupby(by='Sentence length').mean().reset_index()

    desired_order = ['Vertices', 'Generating StanfordNLP representation', 'Generating meuDB', 'Generating intermediate representation', 'Generating logical representation', 'Performing ex post']

    mean_value_columns = [col for col in desired_order if col in df_mean.columns]

    df_melted = pd.melt(df_mean,
                        id_vars=['Sentence length'],
                        value_vars=mean_value_columns,
                        var_name='Metric',
                        value_name='Mean Value')

    df_melted['Metric'] = pd.Categorical(df_melted['Metric'], categories=mean_value_columns, ordered=True)

    all_sentence_lengths = df_melted['Sentence length'].unique()
    all_sentence_lengths.sort()

    plot = (
            ggplot(df_melted, aes(x='Sentence length', y='Mean Value', color='Metric', shape='Metric'))
            + geom_point(size=1.75)
            + geom_line(size=0.75)
            + scale_y_log10(minor_breaks=[],
                          breaks=[10 ** x for x in range(-5, 7)],
                          labels=lambda l: ["{:.0e}".format(v).replace("+0", "+").replace("-0", "-") for v in l])
            + scale_x_continuous(breaks=all_sentence_lengths)
            + labs(title='Mean Values of LaSSI Phases vs. Sentence Length',
                   x='Sentence Length',
                   y='Mean Value (Log Scale, seconds)',
                   color='Metric',
                   shape='Metric')
            + theme_minimal()
            + theme(
                plot_background=element_rect(fill='white', color="white"),
                text=element_text(fontproperties=font_regular),
                legend_text=element_text(ha='left', fontproperties=font_regular),
                axis_title_x=element_text(fontproperties=font_bold),
                axis_title_y=element_text(fontproperties=font_bold),
                legend_title=element_text(ha='left', fontproperties=font_bold),
                plot_title=element_text(ha='center', fontproperties=font_title),
                panel_border=element_blank(),
                legend_position='bottom',
                legend_direction='horizontal'
            )
            + scale_color_brewer(type='qual', palette='Dark2')
            + guides(color=guide_legend(nrow=3), shape=guide_legend(nrow=3))
    )

    plot.save('sentence_length.png', dpi=1200, width=7.5, height=5)
