import streamlit as st
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
from requests import Session
import os
import json
# import helpers
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import timedelta
import datetime


class helpers:
    @staticmethod
    def time_in_utc_530():
        utc_time = datetime.datetime.now(datetime.timezone.utc)
        offset = datetime.timedelta(hours=5, minutes=30)
        time_ = utc_time.astimezone(datetime.timezone(offset))
        return time_.strftime("%H:%M:%S %d-%m-%Y")

    @staticmethod
    def get_session(ncfa):
        new_session = Session()
        new_session.cookies.set("_ncfa", ncfa, domain="www.geoguessr.com")
        return new_session

    @staticmethod
    def get_duel_tokens(session):
        BASE_URL_V4 = "https://www.geoguessr.com/api/v4"
        # only get competitive duels tokens
        game_tokens = []
        pagination_token = None

        def get_token_from_payload(payload):
            try:
                if payload['gameMode'] == 'Duels' and 'competitiveGameMode' in payload:
                    return True
                return False
            except Exception as e:
                return False

        while True:
            response = session.get(
                f"{BASE_URL_V4}/feed/private", params={'paginationToken': pagination_token})
            pagination_token = response.json()['paginationToken']
            entries = response.json()['entries']
            for entry in entries:
                game_date = entry['time']
                game_date = datetime.datetime.fromisoformat(game_date).date()
                start_date = datetime.datetime.strptime(
                    "2024-07-01", "%Y-%m-%d").date()
                if (game_date < start_date):
                    return game_tokens

                payload_json = json.loads(entry['payload'])
                # cleaner way would be to check if payload_json is a dict, if yes
                # then do payload_json=[payload_json]
                # But it's working fine now, after many changes
                # I don't want to change anything lol
                if type(payload_json) is dict:
                    if get_token_from_payload(payload_json):
                        game_tokens.append(payload_json['gameId'])
                else:
                    for payload in payload_json:
                        if (get_token_from_payload(payload['payload'])):
                            game_tokens.append(payload['payload']['gameId'])

            if not pagination_token:
                break
        return game_tokens

    @staticmethod
    def get_player_data(session):
        BASE_URL_V4 = "https://www.geoguessr.com/api/v4"
        try:
            player_data = session.get(
                f"{BASE_URL_V4}/feed/private").json()['entries'][0]['user']
        except:
            return {}
        return {'id': player_data['id'],
                'nick': player_data['nick']}

    @staticmethod
    def get_duels(session, duel_tokens, my_player_Id, loading_bar):
        # add everything to dictionarym then make a dataframe
        data_dict = dict({'Date': [],
                         'Game Id': [],
                          'Round Number': [],
                          'Country': [],
                          'Latitude': [],
                          'Longitude': [],
                          'Damage Multiplier': [],
                          'Opponent Id': [],
                          'Opponent Country': [],
                          'Your Latitude': [],
                          'Your Longitude': [],
                          'Opponent Latitude': [],
                          'Opponent Longitude': [],
                          'Your Distance': [],
                          'Opponent Distance': [],
                          'Your Score': [],
                          'Opponent Score': [],
                          'Map Name': [],
                          'Game Mode': [],
                          'Moving': [],
                          'Zooming': [],
                          'Rotating': [],
                          'Your Rating': [],
                          'Opponent Rating': [],
                          'Score Difference': [],
                          'Win Percentage': [],
                          '5k Border': [],
                          'Pano URL': [],
                          'Pano ID': [],
                          'heading': []
                          })

        BASE_URL_V3 = "https://game-server.geoguessr.com/api/duels"
        count_ = 0
        for token in duel_tokens:
            count_ += 1
            loading_bar.progress(count_/len(duel_tokens))
            response = session.get(f"{BASE_URL_V3}/{token}")
            if response.status_code == 200:
                game = response.json()
                me = 0
                other = 1
                if game['teams'][1]['players'][0]['playerId'] == my_player_Id:
                    me = 1
                    other = 0

                # right now doing the exact same for me and other
                # better way would be to do [me, other] and then loop
                for i in range(game['currentRoundNumber']):
                    round = game['rounds'][i]

                    data_dict['Pano ID'].append(round['panorama']['panoId'])
                    data_dict['heading'].append(round['panorama']['heading'])

                    data_dict['Round Number'].append(round['roundNumber'])
                    data_dict['Country'].append(
                        helpers.get_country_name(round['panorama']['countryCode']))
                    data_dict['Latitude'].append(round['panorama']['lat'])
                    data_dict['Longitude'].append(round['panorama']['lng'])
                    data_dict['Damage Multiplier'].append(
                        round['damageMultiplier'])
                    
                    url_ = "<a href=\"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=" + str(round['panorama']['lat']) + "," + str(round['panorama']['lng']) + "&heading=0&pitch=0&fov=80\">loc link</a>"
                    data_dict['Pano URL'].append(url_)
                    # if no guess is made, there is no entry in guesses of that round, so we find if the round number in round and guess are same, if not, then NAN.
                    my_guess = [guess for guess in game['teams'][me]
                                ['players'][0]['guesses'] if guess['roundNumber'] == i+1]
                    if my_guess:
                        my_guess = my_guess[0]
                        data_dict['Your Latitude'].append(my_guess['lat'])
                        data_dict['Your Longitude'].append(my_guess['lng'])
                        data_dict['Your Distance'].append(
                            my_guess['distance']/1000)
                        data_dict['Your Score'].append(my_guess['score'])
                        if int(my_guess['score']) > 4980:
                            data_dict['5k Border'].append(11)
                        else:
                            data_dict['5k Border'].append(7)
                    else:
                        data_dict['Your Latitude'].append(0)
                        data_dict['Your Longitude'].append(0)
                        data_dict['Your Distance'].append(0)
                        data_dict['Your Score'].append(0)
                        data_dict['5k Border'].append(0)

                    #os.write(1,str.encode(str(my_guess['score'])))
                    
                    

                    other_guess = [guess for guess in game['teams'][other]
                                   ['players'][0]['guesses'] if guess['roundNumber'] == i+1]
                    if other_guess:
                        other_guess = other_guess[0]
                        data_dict['Opponent Latitude'].append(
                            other_guess['lat'])
                        data_dict['Opponent Longitude'].append(
                            other_guess['lng'])
                        data_dict['Opponent Distance'].append(
                            other_guess['distance']/1000)
                        data_dict['Opponent Score'].append(
                            other_guess['score'])
                    else:
                        data_dict['Opponent Latitude'].append(0)
                        data_dict['Opponent Longitude'].append(0)
                        data_dict['Opponent Distance'].append(0)
                        data_dict['Opponent Score'].append(0)
                    data_dict['Score Difference'].append(
                        data_dict['Your Score'][-1] -
                        data_dict['Opponent Score'][-1]
                    )
                    data_dict['Win Percentage'].append(
                        int(data_dict['Your Score'][-1] >
                            data_dict['Opponent Score'][-1])*100
                    )
                    # repeated
                    data_dict['Game Id'].append(game['gameId'])

                    data_dict['Date'].append(game['rounds'][0]['startTime'])

                    data_dict['Map Name'].append(
                        game['options']['map']['name'])
                    data_dict['Game Mode'].append(
                        game['options']['competitiveGameMode'])

                    data_dict['Moving'].append(
                        not game['options']['movementOptions']['forbidMoving'])
                    data_dict['Zooming'].append(
                        not game['options']['movementOptions']['forbidZooming'])
                    data_dict['Rotating'].append(
                        not game['options']['movementOptions']['forbidRotating'])

                    data_dict['Opponent Id'].append(
                        game['teams'][other]['players'][0]['playerId'])
                    data_dict['Opponent Country'].append(helpers.get_country_name(
                        game['teams'][other]['players'][0]['countryCode']))

                    if game['teams'][me]['players'][0]['progressChange'] is not None:
                        # in your placement games, both will be none and the rating will be None
                        if game['teams'][me]['players'][0]['progressChange']['competitiveProgress'] is not None:
                            data_dict['Your Rating'].append(
                                game['teams'][me]['players'][0]['progressChange']['competitiveProgress']['ratingAfter'])
                        else:
                            data_dict['Your Rating'].append(
                                game['teams'][me]['players'][0]['progressChange']["rankedSystemProgress"]['ratingAfter'])
                    else:
                        data_dict['Your Rating'].append(np.nan)
                    # in some cases, both above are none so take just the normal rating
                    if data_dict['Your Rating'][-1] is None:
                        data_dict['Your Rating'][-1] = game['teams'][me]['players'][0]['rating']

                    # some users have progressChange as None and have rating 0, I think they might be in placement stages
                    if game['teams'][other]['players'][0]['progressChange'] is not None:
                        if game['teams'][other]['players'][0]['progressChange']['competitiveProgress'] is not None:
                            data_dict['Opponent Rating'].append(
                                game['teams'][other]['players'][0]['progressChange']['competitiveProgress']['ratingAfter'])
                        else:
                            data_dict['Opponent Rating'].append(
                                game['teams'][other]['players'][0]['progressChange']["rankedSystemProgress"]['ratingAfter'])
                    else:
                        data_dict['Opponent Rating'].append(np.nan)
                    if data_dict['Opponent Rating'][-1] is None:
                        data_dict['Opponent Rating'][-1] = game['teams'][other]['players'][0]['rating']

            else:
                # print(f"Request failed with status code: {response.status_code}")
                # print(f"Response content: {response.text}")
                pass
        return data_dict

    @staticmethod
    def datetime_processing(df):

        def utc_to_offset(series):
            return series + timedelta(hours=5, minutes=30)
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%dT%H:%M:%S.%f%z", errors='coerce').fillna(
            pd.to_datetime(df['Date'], format="%Y-%m-%dT%H:%M:%S%z", errors='coerce'))
        df['Date'] = utc_to_offset(df['Date'])
        df['Time'] = df['Date'].dt.time
        df['Date'] = df['Date'].dt.date
        df['Hour'] = df['Time'].apply(lambda x: x.hour)
        return df

    @staticmethod
    def groupby_country(df):
        by_country = df.groupby('Country').agg({'Your Score': 'mean', 'Opponent Score': 'mean',
                                                'Score Difference': 'mean', 'Win Percentage': 'mean', 'Country': 'count', 'Your Distance': 'mean'})
        by_country.rename(
            columns={'Country': 'Number of Rounds', 'Your Distance': 'Distance'}, inplace=True)
        by_country['Win Percentage'] = by_country['Win Percentage'].apply(
            lambda x: round(x, 2))
        by_country[['Your Score', 'Opponent Score', 'Score Difference', 'Distance']] = by_country[[
            'Your Score', 'Opponent Score', 'Score Difference', 'Distance']].apply(round)

        new_cols = ['Number of Rounds'] + \
            [col for col in by_country.columns if col != 'Number of Rounds']
        by_country = by_country[new_cols]
        return by_country

    @staticmethod
    def get_country_name(country_code):
        country_code = country_code.lower()
        country_name_dict = {'ad': 'Andorra',
                             'ae': 'United Arab Emirates',
                             'af': 'Afghanistan',
                             'ag': 'Antigua and Barbuda',
                             'ai': 'Anguilla',
                             'al': 'Albania',
                             'am': 'Armenia',
                             'ao': 'Angola',
                             'aq': 'Antarctica',
                             'ar': 'Argentina',
                             'as': 'American Samoa',
                             'at': 'Austria',
                             'au': 'Australia',
                             'aw': 'Aruba',
                             'ax': 'Åland Islands',
                             'az': 'Azerbaijan',
                             'ba': 'Bosnia and Herzegovina',
                             'bb': 'Barbados',
                             'bd': 'Bangladesh',
                             'be': 'Belgium',
                             'bf': 'Burkina Faso',
                             'bg': 'Bulgaria',
                             'bh': 'Bahrain',
                             'bi': 'Burundi',
                             'bj': 'Benin',
                             'bl': 'Saint Barthélemy',
                             'bm': 'Bermuda',
                             'bn': 'Brunei Darussalam',
                             'bo': 'Bolivia',
                             'bq': 'Bonaire, Sint Eustatius and Saba',
                             'br': 'Brazil',
                             'bs': 'Bahamas',
                             'bt': 'Bhutan',
                             'bv': 'Bouvet Island',
                             'bw': 'Botswana',
                             'by': 'Belarus',
                             'bz': 'Belize',
                             'ca': 'Canada',
                             'cc': 'Cocos (Keeling) Islands',
                             'cd': 'Congo (Democratic Republic of the)',
                             'cf': 'Central African Republic',
                             'cg': 'Congo',
                             'ch': 'Switzerland',
                             'ci': 'Côte d\'Ivoire',
                             'ck': 'Cook Islands',
                             'cl': 'Chile',
                             'cm': 'Cameroon',
                             'cn': 'China',
                             'co': 'Colombia',
                             'cr': 'Costa Rica',
                             'cu': 'Cuba',
                             'cv': 'Cabo Verde',
                             'cw': 'Curaçao',
                             'cx': 'Christmas Island',
                             'cy': 'Cyprus',
                             'cz': 'Czechia',
                             'de': 'Germany',
                             'dj': 'Djibouti',
                             'dk': 'Denmark',
                             'dm': 'Dominica',
                             'do': 'Dominican Republic',
                             'dz': 'Algeria',
                             'ec': 'Ecuador',
                             'ee': 'Estonia',
                             'eg': 'Egypt',
                             'eh': 'Western Sahara',
                             'er': 'Eritrea',
                             'es': 'Spain',
                             'et': 'Ethiopia',
                             'fi': 'Finland',
                             'fj': 'Fiji',
                             'fk': 'Falkland Islands (Malvinas)',
                             'fm': 'Micronesia (Federated States of)',
                             'fo': 'Faroe Islands',
                             'fr': 'France',
                             'ga': 'Gabon',
                             'gb': 'United Kingdom',
                             'gd': 'Grenada',
                             'ge': 'Georgia',
                             'gf': 'French Guiana',
                             'gg': 'Guernsey',
                             'gh': 'Ghana',
                             'gi': 'Gibraltar',
                             'gl': 'Greenland',
                             'gm': 'Gambia',
                             'gn': 'Guinea',
                             'gp': 'Guadeloupe',
                             'gq': 'Equatorial Guinea',
                             'gr': 'Greece',
                             'gs': 'South Georgia and the South Sandwich Islands',
                             'gt': 'Guatemala',
                             'gu': 'Guam',
                             'gw': 'Guinea-Bissau',
                             'gy': 'Guyana',
                             'hk': 'Hong Kong',
                             'hm': 'Heard Island and McDonald Islands',
                             'hn': 'Honduras',
                             'hr': 'Croatia',
                             'ht': 'Haiti',
                             'hu': 'Hungary',
                             'id': 'Indonesia',
                             'ie': 'Ireland',
                             'il': 'Israel',
                             'im': 'Isle of Man',
                             'in': 'India',
                             'io': 'British Indian Ocean Territory',
                             'iq': 'Iraq',
                             'ir': 'Iran',
                             'is': 'Iceland',
                             'it': 'Italy',
                             'je': 'Jersey',
                             'jm': 'Jamaica',
                             'jo': 'Jordan',
                             'jp': 'Japan',
                             'ke': 'Kenya',
                             'kg': 'Kyrgyzstan',
                             'kh': 'Cambodia',
                             'ki': 'Kiribati',
                             'km': 'Comoros',
                             'kn': 'Saint Kitts and Nevis',
                             'kp': 'North Korea',
                             'kr': 'South Korea',
                             'kw': 'Kuwait',
                             'ky': 'Cayman Islands',
                             'kz': 'Kazakhstan',
                             'la': 'Laos',
                             'lb': 'Lebanon',
                             'lc': 'Saint Lucia',
                             'li': 'Liechtenstein',
                             'lk': 'Sri Lanka',
                             'lr': 'Liberia',
                             'ls': 'Lesotho',
                             'lt': 'Lithuania',
                             'lu': 'Luxembourg',
                             'lv': 'Latvia',
                             'ly': 'Libya',
                             'ma': 'Morocco',
                             'mc': 'Monaco',
                             'md': 'Moldova',
                             'me': 'Montenegro',
                             'mf': 'Saint Martin',
                             'mg': 'Madagascar',
                             'mh': 'Marshall Islands',
                             'mk': 'North Macedonia',
                             'ml': 'Mali',
                             'mm': 'Myanmar',
                             'mn': 'Mongolia',
                             'mo': 'Macao',
                             'mp': 'Northern Mariana Islands',
                             'mq': 'Martinique',
                             'mr': 'Mauritania',
                             'ms': 'Montserrat',
                             'mt': 'Malta',
                             'mu': 'Mauritius',
                             'mv': 'Maldives',
                             'mw': 'Malawi',
                             'mx': 'Mexico',
                             'my': 'Malaysia',
                             'mz': 'Mozambique',
                             'na': 'Namibia',
                             'nc': 'New Caledonia',
                             'ne': 'Niger',
                             'nf': 'Norfolk Island',
                             'ng': 'Nigeria',
                             'ni': 'Nicaragua',
                             'nl': 'Netherlands',
                             'no': 'Norway',
                             'np': 'Nepal',
                             'nr': 'Nauru',
                             'nu': 'Niue',
                             'nz': 'New Zealand',
                             'om': 'Oman',
                             'pa': 'Panama',
                             'pe': 'Peru',
                             'pf': 'French Polynesia',
                             'pg': 'Papua New Guinea',
                             'ph': 'Philippines',
                             'pk': 'Pakistan',
                             'pl': 'Poland',
                             'pm': 'Saint Pierre and Miquelon',
                             'pn': 'Pitcairn',
                             'pr': 'Puerto Rico',
                             'ps': 'Palestine',
                             'pt': 'Portugal',
                             'pw': 'Palau',
                             'py': 'Paraguay',
                             'qa': 'Qatar',
                             're': 'Réunion',
                             'ro': 'Romania',
                             'rs': 'Serbia',
                             'ru': 'Russia',
                             'rw': 'Rwanda',
                             'sa': 'Saudi Arabia',
                             'sb': 'Solomon Islands',
                             'sc': 'Seychelles',
                             'sd': 'Sudan',
                             'se': 'Sweden',
                             'sg': 'Singapore',
                             'sh': 'Saint Helena',
                             'si': 'Slovenia',
                             'sj': 'Svalbard and Jan Mayen',
                             'sk': 'Slovakia',
                             'sl': 'Sierra Leone',
                             'sm': 'San Marino',
                             'sn': 'Senegal',
                             'so': 'Somalia',
                             'sr': 'Suriname',
                             'ss': 'South Sudan',
                             'st': 'Sao Tome and Principe',
                             'sv': 'El Salvador',
                             'sx': 'Sint Maarten',
                             'sy': 'Syria',
                             'sz': 'Eswatini',
                             'tc': 'Turks and Caicos Islands',
                             'td': 'Chad',
                             'tf': 'French Southern Territories',
                             'tg': 'Togo',
                             'th': 'Thailand',
                             'tj': 'Tajikistan',
                             'tk': 'Tokelau',
                             'tl': 'Timor-Leste',
                             'tm': 'Turkmenistan',
                             'tn': 'Tunisia',
                             'to': 'Tonga',
                             'tr': 'Turkey',
                             'tt': 'Trinidad and Tobago',
                             'tv': 'Tuvalu',
                             'tw': 'Taiwan',
                             'tz': 'Tanzania',
                             'ua': 'Ukraine',
                             'ug': 'Uganda',
                             'um': 'United States Minor Outlying Islands',
                             'us': 'United States',
                             'uy': 'Uruguay',
                             'uz': 'Uzbekistan',
                             'va': 'Vatican City',
                             'vc': 'Saint Vincent and the Grenadines',
                             've': 'Venezuela',
                             'vg': 'British Virgin Islands',
                             'vi': 'U.S. Virgin Islands',
                             'vn': 'Vietnam',
                             'vu': 'Vanuatu',
                             'wf': 'Wallis and Futuna',
                             'ws': 'Samoa',
                             'xk': 'Kosovo',
                             'ye': 'Yemen',
                             'yt': 'Mayotte',
                             'za': 'South Africa',
                             'zm': 'Zambia',
                             'zw': 'Zimbabwe', }
        if country_code in country_name_dict.keys():
            return country_name_dict[country_code]
        else:
            return country_code

    @staticmethod
    def alt_chart(data, x, y):
        c = alt.Chart(data).mark_bar().encode(x=alt.X(x, sort=None), y=y)
        st.altair_chart(c)

    @staticmethod
    def create_line_chart(df,  metric_col, date_option):
        date_col = 'Date'

        df['Date'] = pd.to_datetime(df['Date'])
        group_options = {
            'Week': 'W',
            'Month': 'M',
            'Year': 'Y'
        }

        df['Group'] = df[date_col].dt.to_period(
            group_options[date_option]).apply(lambda r: r.start_time)
        df_grouped = df.groupby(by='Group')[metric_col].mean()

        fig = px.line(df_grouped,  y=metric_col, markers=True)
        fig.update_layout(bargap=0.1, xaxis_title=date_option,
                          yaxis_title=metric_col)
        fig.update_layout(title_text=f"{metric_col}")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def create_line_chart_games_played(df,  date_option):
        date_option = st.radio(
            'A', ('Week', 'Month', 'Year'), horizontal=True, label_visibility='hidden')
        date_col = 'Date'
        metric_col = 'Games Played'
        df[metric_col] = df['Game Id']
        df['Date'] = pd.to_datetime(df['Date'])

        group_option = {
            'Week': 'W',
            'Month': 'M',
            'Year': 'Y'
        }
        df.loc[:, 'Group'] = df[date_col].dt.to_period(
            group_option[date_option]).apply(lambda r: r.start_time)
        df_grouped = df.groupby(by='Group')[metric_col].nunique()

        fig = px.line(df_grouped,  y=metric_col, markers=True)
        fig.update_layout(xaxis_title=date_option, yaxis_title=metric_col)
        # fig.update_layout(title_text=f"Games Played")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def scatter_by_game_type(top_n_countries, df, col_a, col_b, metric_col, show_avg_lines, color=None):

        df = df[df['Country'].isin(top_n_countries.index)]

        df_a = df if col_a == 'Moving' else df[~df['Moving']] if col_a == 'No Move' else df[(
            ~df['Moving']) & (~df['Zooming'])]
        df_b = df if col_b == 'Moving' else df[~df['Moving']] if col_b == 'No Move' else df[(
            ~df['Moving']) & (~df['Zooming'])]

        if metric_col == 'Number of Rounds':
            metric_col = 'Round Number'
            df_a = df_a.groupby('Country')[metric_col].count()
            df_b = df_b.groupby('Country')[metric_col].count()
        else:
            if metric_col == 'Distance':
                metric_col = 'Your Distance'
            df_a = df_a.groupby('Country')[metric_col].mean()
            df_b = df_b.groupby('Country')[metric_col].mean()

        if col_a == col_b:
            col_a = col_a+' A'
            col_b = col_b+' B'

        df_a.rename(col_a, inplace=True)
        df_b.rename(col_b, inplace=True)
        df = pd.concat([df_a, df_b], axis=1)

        labels_ = df.index.map(
            lambda x: x if df.index.get_loc(x) % 5 == 0 else "")
        fig = px.scatter(data_frame=df.reset_index(), x=col_a, y=col_b, text=labels_,
                         hover_name='Country', color=color, color_continuous_scale='RdBu')
        fig.update_layout(coloraxis_colorbar_title_side='bottom')
        fig.update_coloraxes(colorbar_title_text='',
                             colorbar_xpad=0, colorbar_thickness=5)
        if (show_avg_lines):
            fig.add_shape(
                type="line",
                x0=df[col_a].min(), y0=df[col_b].mean(), x1=df[col_a].max(), y1=df[col_b].mean(),
                line=dict(color='green', width=2, dash="dot"),
                xref="x", yref="y"
            )
            # vertical line
            fig.add_shape(
                type="line",
                x0=df[col_a].mean(), y0=df[col_b].min(), x1=df[col_a].mean(), y1=df[col_b].max(),
                line=dict(color='green', width=2, dash="dot"),
                xref="x", yref="y"
            )
        fig.update_traces(textposition='top center')
        # fig.update_layout(yaxis_range=[0,5000])
        # fig.update_layout(xaxis_range=[0,5000])
        size_ = 600
        fig.update_layout(width=size_, height=size_)
        st.plotly_chart(fig, use_container_width=False)




st.title('Welcome to Duels Analyzer')
st.text('I created this tool to analyse my rated duel games on Geoguessr. I hope you find it helpful.')
st.text('It needs your _ncfa token to get your games history and data. Your token is not sent anywhere neither it is saved anywhere. You can check the source code, it is open source.')
with st.form('input_token'):
    _ncfa = st.text_input("Enter _ncfa token:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button('How to get your _ncfa token',
                       "https://github.com/SafwanSipai/geo-insight?tab=readme-ov-file#getting-your-_ncfa-cookie")
    with col3:
        submitted_token = st.form_submit_button("Enter")

if 'submitted_token' not in st.session_state:
    st.session_state['submitted_token'] = False

if (submitted_token or st.session_state['submitted_token']) and _ncfa:
    st.session_state['submitted_token'] = True
    geoguessr_session = helpers.get_session(_ncfa)
    player_data = helpers.get_player_data(geoguessr_session)
    if player_data != {}:
        my_player_Id = player_data['id']
        st.write(
            f"Hello {player_data['nick']} (id {player_data['id']}), extracting your game tokens...")
        print(helpers.time_in_utc_530(),
              player_data['nick'], player_data['id'])
    if 'duel_tokens' not in st.session_state:
        st.session_state['duel_tokens'] = []
        with st.spinner("", show_time=True):
            duel_tokens = helpers.get_duel_tokens(geoguessr_session)
        st.session_state['duel_tokens'] = duel_tokens
    else:
        duel_tokens = st.session_state['duel_tokens']
    st.write(f"Found {len(duel_tokens)} rated duels.")

    st.write(
        f"To retrive all {len(duel_tokens)} games, it will take around {60*len(duel_tokens)/500} seconds.")
    st.markdown('I recommend you choose **All**, it will take some time but after that, you can analyse your games withouth any loading.')
    retrieval_option = st.radio(
        "Retrieval Option:",
        # ("Retrieve All", "Retrieve Recent", "Retrieve by Date"),
        ("Retrieve All", "Retrieve Recent"),
        horizontal=False,
        label_visibility="collapsed",
    )
    with st.form("retrieval_form", border=False):
        if retrieval_option == "Retrieve Recent":
            recent_count = st.slider("Recent Games:", 1, len(
                duel_tokens), round(len(duel_tokens)/2))
        # elif retrieval_option == "Retrieve by Date":
        #     today = datetime.date.today()
        #     start_date = today - datetime.timedelta(days=7)
        #     date_range = st.date_input("Select a date range", (start_date, today),format="DD/MM/YYYY")
        else:
            recent_count = None
            date_range = None
        submitted_1 = st.form_submit_button("Retrieve")
    if 'submitted_1' not in st.session_state:
        st.session_state['submitted_1'] = False
    if st.session_state['submitted_1'] or submitted_1:
        st.session_state['submitted_1'] = True
        if retrieval_option == "Retrieve All":
            st.write("Retrieving all games' data...")
        elif retrieval_option == "Retrieve Recent":
            st.write(f"Retrieving {recent_count} recent games...")
            duel_tokens = duel_tokens[:recent_count]
        # else:
            # st.write(f"Retrieving games between {date_range[0]} and {date_range[1]}...")
            # to do the whole retrival  by date thing
        data_dict = {}
        geoguessr_session_2 = Session()
        geoguessr_session_2.cookies.set(
            "_ncfa", _ncfa, domain=".geoguessr.com")
        if len(duel_tokens) > 0:
            if 'data_dict' not in st.session_state:
                st.session_state['data_dict'] = {}
                loading_bar = st.progress(0)
                data_dict = helpers.get_duels(
                    geoguessr_session_2, duel_tokens, my_player_Id, loading_bar)
                st.success('Done')
                st.session_state['data_dict'] = data_dict
            else:
                data_dict = st.session_state['data_dict']
                st.success('Done')
        df = pd.DataFrame()
        df = pd.DataFrame(data_dict)
        if not df.empty:
            df = helpers.datetime_processing(df)
        # st.write(df)
        submitted = False
        option = st.radio(
            'How many games you want to analyze?',
            ('All', 'Recent games', 'By Date'))
        if option == 'Recent games':
            with st.form("option_form"):
                slider_value = st.slider("Select how many recent games you want to analyse:",
                                         min_value=1, max_value=len(duel_tokens), value=len(duel_tokens))
                submitted = st.form_submit_button("Submit")

        elif option == 'By Date':
            with st.form("option_form"):
                today = datetime.date.today()
                start_date = today - datetime.timedelta(days=7)
                date_range = st.date_input(
                    "Select a date range", (start_date, today), format="DD/MM/YYYY")
                submitted = st.form_submit_button("Submit")
        else:
            with st.form("option_form"):
                submitted = st.form_submit_button("Submit")
        if 'submitted' not in st.session_state:
            st.session_state['submitted'] = False
        if (st.session_state['submitted'] or submitted) and not df.empty:
            st.session_state['submitted'] = True
            df_filtered = pd.DataFrame()
            if option == 'By Date':
                df_filtered = df[(df['Date'] >= date_range[0]) & (
                    df['Date'] <= date_range[1])].copy()
                st.write(f"Found {df_filtered['Game Id'].nunique()} games")
            elif option == 'Recent games':
                df['Running Total'] = (
                    df['Game Id'] != df['Game Id'].shift()).cumsum()
                df_filtered = df[df['Running Total'] <= slider_value].copy()
            else:
                df_filtered = df.copy()
            by_country = helpers.groupby_country(df_filtered)
            top_n = st.slider('Select how many countries you want to see (by round count):', min_value=1, max_value=len(
                by_country), value=20, step=1, help='This helps filter out the countries that occur very rarely.')
            top_n_countries = by_country.sort_values(
                by='Number of Rounds', ascending=False).head(top_n)

            if not df_filtered.empty:
                st.markdown('### Detailed Analysis')
                with st.expander(""):
                    metric = st.radio(
                        'Choose a metric:', ('Score', 'Distance', 'Score Difference', 'Win Percentage'))
                    if metric == 'Score':
                        metric_col = 'Your Score'
                    else:
                        metric_col = metric

                    gtype = st.radio(
                        'Choose a game Type:', ('No Move', 'Moving', 'NMPZ'))
                    if gtype == 'Moving':
                        gametype = 'StandardDuels'
                    elif gtype == 'No Move':
                        gametype = 'NoMoveDuels'
                    else:
                        gametype = 'NmpzDuels'

                    st.markdown(f"#### All your guesses, colored by {metric}")
                    
                    st.write(f"\t{metric_col} %")
                    lat_col = df_filtered['Latitude'].loc[df_filtered['Game Mode'] == gametype]
                    lon_col = df_filtered['Longitude'].loc[df_filtered['Game Mode'] == gametype]
                    color_ = {"sequential": [
                    [0, 'rgb(191, 34, 34)'], 
                    [0.3, 'rgb(243, 10, 10)'],
                    [0.5, 'rgb(234, 174, 19)'],
                    [0.75, 'rgb(220, 231, 22)'],
                    [0.85, 'rgb(26, 227, 40)'],
                    [0.90, 'rgb(34, 187, 175)'],
                    [0.95, 'rgb(24, 111, 197)'],
                    [0.995, 'rgb(47, 47, 255)'],
                    [0.996, 'rgb(255, 255, 255)'],
                    [1, 'rgb(255, 255, 255)']
                    ]}

                    if metric_col == 'Distance':
                        metric_col = 'Your Distance'
                    if metric_col == 'Your Score':
                        color_ = {"sequential": [
                        [0, 'rgb(191, 34, 34)'], 
                        [0.3, 'rgb(243, 10, 10)'],
                        [0.5, 'rgb(234, 174, 19)'],
                        [0.75, 'rgb(220, 231, 22)'],
                        [0.85, 'rgb(26, 227, 40)'],
                        [0.90, 'rgb(34, 187, 175)'],
                        [0.95, 'rgb(24, 111, 197)'],
                        [0.995, 'rgb(47, 47, 255)'],
                        [0.996, 'rgb(255, 255, 255)'],
                        [1, 'rgb(255, 255, 255)']
                        ]}
                    if metric_col == 'Your Distance':
                        color_ = {"sequential": [
                                [0, 'rgb(255, 255, 255)'], 
                                [0.00025, 'rgb(255, 255, 255)'],
                                [0.000255, 'rgb(85, 9, 213)'],
                                [0.0025, 'rgb(35, 31, 119)'],
                                [0.005, 'rgb(20, 102, 212)'],
                                [0.025, 'rgb(17, 155, 166)'],
                                [0.05, 'rgb(24, 111, 197)'],
                                [0.1, 'rgb(47, 47, 255)'],
                                [0.25, 'rgb(255, 255, 255)'],
                                [1, 'rgb(255, 255, 255)']
                                ]}
                    if metric_col == 'Score Difference':
                        color_ = {"sequential": [
                                [0, 'rgb(242, 255, 0)'], 
                                [0.2, 'rgb(242, 255, 0)'],
                                [0.4, 'rgb(242, 255, 0)'],
                                [0.45, 'rgb(242, 255, 0)'],
                                [0.475, 'rgb(242, 255, 0)'],
                                [0.490, 'rgb(242, 255, 0)'],
                                [0.5, 'rgb(242, 255, 0)'],
                                [0.510, 'rgb(255, 0, 233)'],
                                [0.525, 'rgb(255, 0, 233)'],
                                [0.55, 'rgb(255, 0, 233)'],
                                [0.6, 'rgb(255, 0, 233)'],
                                [0.8, 'rgb(255, 0, 233)'],
                                [1, 'rgb(255, 0, 233)']
                                ]}
                    fig = go.Figure()

                    app = dash.Dash(__name__)

                    app.layout = html.Div([
                    dcc.Graph(
                    id='scatter-plot',
                    config={'displayModeBar': False},
                    figure = fig
                    )
                    ])

                    fig.add_trace(go.Scattermap(
                    lat=lat_col,
                    lon=lon_col,
                    mode='markers',
                    marker=go.scattermap.Marker(
                            size=df_filtered["5k Border"],
                            color="Black"
                        ),
                        text=df_filtered['Pano URL'],
                        hoverinfo='text'
                        ))
                    
                    fig.add_trace(go.Scattermap(
                        lat=lat_col,
                        lon=lon_col,
                        mode='markers',
                        marker=go.scattermap.Marker(
                        size=6,
                        color=df_filtered[metric_col]
                        ),
                        text=df_filtered['Pano URL'],
                        hoverinfo='text'
                        ))

                    fig.update_layout(
                        title=dict(text='Your guesses:'),
                        autosize=True,
                        hovermode='closest',
                        showlegend=False,
                        colorscale=color_,
                        map=dict(
                        bearing=0,
                        pitch=0,
                        zoom=0,
                        style='light'
                        ),
                        )

                    if metric_col == 'Your Distance':
                            fig.update_layout(coloraxis=dict(cmin=0, cmax=20000))
                    if metric_col == 'Score Difference':
                            fig.update_layout(coloraxis=dict(cmin=-5000, cmax=5000))

                    fig.update_layout(map_style="open-street-map")
                    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
                    st.plotly_chart(fig)


                    @app.callback(
                        Output('scatter-plot', 'figure'),
                        Input('scatter-plot', 'clickData')
                    )
                    def update_scatter_plot(clickData):
                        fig.update_traces(hovertemplate='Click me!<extra></extra>')
                    # Add custom hover text for each point
                        return fig

                    @app.callback(
                        Output('scatter-plot', 'clickData'),
                        Input('scatter-plot', 'clickData')
                    )
                    def display_click_data(clickData):
                        if clickData:
                            point_index = clickData['points'][0]['pointIndex']
                        url = df['Pano URL'][point_index]
                        # Open the URL in a new tab
                        import webbrowser
                        webbrowser.open(url)
                    
                        return clickData

                    #if __name__ == '__main__':
                    #    app.run_server(debug=True, use_reloader=False, port=4444)
                    #### Complete extracted data (Download for your own analysis)')
                    st.write(df_filtered)

                    badlocs = pd.DataFrame()
                    badlocs=df[df['Your Score'].between(0, 3000)]

                    json_ = "{\"name\":\"test\",\"customCoordinates\":["

                    for index, row in badlocs.iterrows():
                        loc_ = "{\"lat\":" + str(row['Latitude']) + ",\"lng\":" +  str(row['Longitude']) + ",\"pitch\":0,\"zoom\":0,\"heading\":" + str(row['heading']) + ",\"panoId\":null,\"countryCode\":null,\"stateCode\":null,\"extra\":{\"tags\":[]}},"
                        json_ = json_ + loc_

                    json_ = json_ + "]}"

                    if st.button("Prepare download"):
                        st.download_button(
                            label="Download text",
                            data=json_,
                            file_name="locations.json",
                            on_click="ignore",
                            type="primary",
                            icon=":material/download:",
                        )
