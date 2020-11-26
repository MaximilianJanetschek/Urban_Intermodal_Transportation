import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.spatial import distance




'''
README: 
This is a standalone file, run to execute
Requirement: Please make sure in results/distances folder:
1. results0 includes the BASELINE beta set 0 (100 requests)
2. results1 includes the BIASED beta set 1 (100 requests)
PLEASE UNCOMMENT THE GRAPH DRAWING PART IF YOU WANT TO SEE THE GRAPH

'''

def inputcsv(filenumber):
    """Import the result request run for each beta csv file and include the column
    of beta set to classify which beta set the data belong to.
    Output: Data Frame"""
    # Import request run result for a beta set
    with open(r'results/distances/StatisticalAnalysis/results' + str(filenumber) + '.CSV', newline='') as csvfile:
        df_reader = pd.read_csv(csvfile)
    # Add new information columns to the existing data frame
    df_reader['betaSet'] = filenumber
    df_reader['ED.x'] = df_reader['cost']
    df_reader['ED.y'] = df_reader['ptTime'] + df_reader['taxiTime'] + df_reader['bikeTime'] + \
                        df_reader['walkingTime'] + df_reader['waitingTime']
    df_reader['totalTime'] = df_reader['ptTime'] + df_reader['taxiTime'] + df_reader['bikeTime'] + \
                        df_reader['walkingTime'] + df_reader['waitingTime']
    df_reader['outVehicleTime'] = df_reader['walkingTime'] + df_reader['waitingTime']
    df_reader['PhysicalTime'] = df_reader['walkingTime'] + df_reader['bikeTime']
    #turn cost into categorical column
    df_reader['CostType'] = 'WalkOnly'
    df_reader.loc[(df_reader['cost'] ==1.5),'CostType'] = 'Bike'
    df_reader.loc[(df_reader['cost'] == 2.9), 'CostType'] = 'pt'
    df_reader.loc[(df_reader['cost'] == 4.4), 'CostType'] = 'Bike+pt'
    df_reader.loc[(df_reader['cost'] > 4.4), 'CostType'] = 'Taxi'



    return df_reader

def mergeinput_preprocess_duo(df1, df2):
    """ Merge 2 data frames into one for full analysis.
    Output will be two form: long form and wide form
    long form: simply add 2 df on top of each other
    wide form: each row represents a request coordinate with result from both run. -> calculate ED
    relevant cols for wide form: maxChanges, origin, destination, ED.x1,ED.y1,ED.x2,ED.y2,ED.x1.normalized,ED.x2.normalized
    ED"""
    ## long form
    df_long = pd.concat([df1,df2],ignore_index = True)

    #normalize the x and y coordinates
    df_long['ED.x.normalized'] = (df_long['ED.x'] - df_long['ED.x'].min()) / (df_long['ED.x'].max() - df_long['ED.x'].min())
    df_long['ED.y.normalized'] = (df_long['ED.y'] - df_long['ED.y'].min()) / (
                df_long['ED.y'].max() - df_long['ED.y'].min())
    ## wide form
    df_temp = df_long.drop(
        columns=['beta_1', 'beta_2', 'beta_3', 'beta_4', 'beta_5', 'Disutility', 'inVehicleTime', 'ptTime',
                 'taxiTime', 'bikeTime', 'walkingTime', 'waitingTime', 'cost',
                 'numberOfChanges', 'preprocessingTime', 'optimizationTime', 'Gap'])
    betaSetx = df_temp['betaSet'][0]
    betaSety = df_temp['betaSet'][199]
    df_temp0 = df_temp[df_temp['betaSet'] == betaSetx]
    df_temp0 =df_temp0.drop(columns='betaSet')
    df_temp1 = df_temp[df_temp['betaSet'] == betaSety]
    df_temp1 = df_temp1.drop(columns='betaSet')
    df_wide = pd.merge(df_temp0, df_temp1, on=('origin','destination','maxChanges'), suffixes=('_base','_mod'))
    df_wide['ED'] = np.sqrt(np.square(df_wide['ED.x.normalized_base']-df_wide['ED.x.normalized_mod'])
                            + np.square(df_wide['ED.y.normalized_base']-df_wide['ED.y.normalized_mod']))
    df_wide['request_no'] = range(1, len(df_wide) + 1)
    return df_long, df_wide

def mergeinput(df1, df2):
    ## long form
    df_long = pd.concat([df1,df2],ignore_index = True)
    return df_long

def normalize(df_long):
    '''Normalize the values and output df_long and df_wide'''
    #normalize the x and y coordinates
    df_long['ED.x.normalized'] = (df_long['ED.x'] - df_long['ED.x'].min()) / (df_long['ED.x'].max() - df_long['ED.x'].min())
    df_long['ED.y.normalized'] = (df_long['ED.y'] - df_long['ED.y'].min()) / (
                df_long['ED.y'].max() - df_long['ED.y'].min())
    return df_long

def splitdf(df_long,i):
    ## wide form
    df_temp = df_long.drop(
        columns=['beta_1', 'beta_2', 'beta_3', 'beta_4', 'beta_5', 'Disutility', 'inVehicleTime', 'ptTime',
                 'taxiTime', 'bikeTime', 'walkingTime', 'waitingTime', 'cost',
                 'numberOfChanges', 'preprocessingTime', 'optimizationTime', 'Gap'])
    betaSetx = df_temp['betaSet'][0]
    betaSety = df_temp['betaSet'][i*100]
    df_temp0 = df_temp[df_temp['betaSet'] == betaSetx]
    df_temp0 =df_temp0.drop(columns='betaSet')
    df_temp1 = df_temp[df_temp['betaSet'] == betaSety]
    df_temp1 = df_temp1.drop(columns='betaSet')
    df_wide = pd.merge(df_temp0, df_temp1, on=('origin','destination','maxChanges'), suffixes=('_0','_1'))
    df_wide['ED'] = np.sqrt(np.square(df_wide['ED.x.normalized_0']-df_wide['ED.x.normalized_1'])
                            + np.square(df_wide['ED.y.normalized_0']-df_wide['ED.y.normalized_1']))
    df_wide['request_no'] = range(1, len(df_wide) + 1)
    return df_wide



def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements.
    Input: DF[column] for the desired column """
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

def inputrequest(jsonFileNumber):
    """ Input the request coordinates to the dataframe. THE ORIGINAL ORDER MUST BE INTACT"""
    df_reader = pd.read_json(r'data/requests/' + str(jsonFileNumber) + 'requests.json')
    df_reader['request_no'] = range(1, len(df_reader) + 1)
    df_reader['ori-destED'] = [distance.euclidean([df_reader['fromLat'][i],df_reader['fromLon'][i]],
                                                  [df_reader['toLat'][i],df_reader['toLon'][i]])
                               for i in range(0,len(df_reader['request_no']))]
    return df_reader

def addEDInfo(df_wide,df_request):
    """Add coordination information of requests into result data frame"""
    df = pd.merge(df_wide,df_request, on='request_no')
    return df

def CItest(df,alpha):
    """ test whether the lower range of CI is larger than 0 or not"""
    ci = norm(*norm.fit(df['ED'])).interval(alpha)
    print(ci)
    if ci[0] < 0:
        print(str(alpha*100)+"%","Confidence: ED not significantly different from 0")
    else:
        print(str(alpha*100)+"%","Confidence: ED significantly different from 0")
    return

# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l=='WalkOnly':
            cols.append('tab:green')
            # cols.append('red')
        elif l == 'Bike':
            cols.append('tab:blue')
            # cols.append('orange')
        elif l == 'pt':
            cols.append('tab:purple')
            # cols.append('purple')
        elif l == 'Bike+pt':
            cols.append('tab:orange')
            # cols.append('orange')
        else:
            # cols.append('tab:red')
            cols.append('red')

    colslabel = {'tab:green':'Walk Only',
                 'tab:blue':'Bike',
                 'tab:purple':'pt',
                 'tab:orange':'Bike+pt',
                 'tab:red':'Taxi'}
    return colslabel,cols

def compare(df_base,df1):
    """ Compare the base beta set with the current beta set and only keep the distinct route"""
    df_base_temp = df_base.drop(
        columns=['ED.x','ED.y','preprocessingTime', 'optimizationTime', 'Gap'])
    df1_temp = df1.drop(
        columns=['ED.x','ED.y','Disutility',
                  'preprocessingTime', 'optimizationTime', 'Gap'])
    df_wide = pd.merge(df_base_temp, df1_temp, on=('origin', 'destination', 'maxChanges'), suffixes=('_base', '_set{}'.format(str(df1['betaSet'][1]))))
    # Compare the corresponding route from 2 beta sets
    df_wide['DistinctRoute?'] = [False if (abs(df_wide['inVehicleTime_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['inVehicleTime_base'][i]) < 0.0001
                                          and
                                          abs(df_wide['ptTime_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['ptTime_base'][i]) < 0.0001
                                          and
                                          abs(df_wide['outVehicleTime_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['outVehicleTime_base'][i]) < 0.0001
                                          and
                                          abs(df_wide['bikeTime_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['bikeTime_base'][i]) < 0.0001
                                          and
                                          abs(df_wide['walkingTime_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['walkingTime_base'][i]) < 0.0001
                                          and
                                          abs(df_wide['waitingTime_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['waitingTime_base'][i]) < 0.0001
                                          and
                                          abs(df_wide['cost_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['cost_base'][i]) < 0.0001
                                          # and
                                          # abs(df_wide['numberOfChanges_set{}'.format(str(df1['betaSet'][1]))][i] - df_wide['numberOfChanges_base'][i]) < 0.0001

                                          )else True for i in range(0,len(df_wide))]

    return df_wide

def extractdistinct(df_wide):
    """Extract and reformat to standard form for disinct point set"""
    df1_distincttemp = df_wide[df_wide['DistinctRoute?'] == True]
    # df1_distinct = df1_distinct.drop([x for x in df1_distinct if x.endswith('_base')], 1)

    df1_distinct = df1_distincttemp.reset_index(drop=True)
    return df1_distinct, df1_distincttemp

def drawline(betaset,df1_distinct,j,x_axis,y_axis,linecolor):
    """Draw a line between points from different beta sets that share the same ori-dest coordination"""
    plt.plot([df1_distinct['{}_base'.format(x_axis)][j],df1_distinct['{}_set{}'.format(x_axis,betaset)][j]],
             [df1_distinct['{}_base'.format(y_axis)][j],df1_distinct['{}_set{}'.format(y_axis,betaset)][j]]
             ,color=linecolor,linestyle='--',linewidth=0.4,marker='.',markersize=0.000001)

    return

def calculateED(betaset,df1_distinct_oldpos,x_axis,y_axis):
    df1_distinct_oldpos['ED'] = np.sqrt(np.square(df1_distinct_oldpos['{}_base'.format(x_axis)]-df1_distinct_oldpos['{}_set{}'.format(x_axis,betaset)])
                            + np.square(df1_distinct_oldpos['{}_base'.format(y_axis)]-df1_distinct_oldpos['{}_set{}'.format(y_axis,betaset)]))
    return df1_distinct_oldpos

######################### Execution  #######################################
"""
README: Adjust the end_range depend on the number of beta sets run
"""


df_base = inputcsv(0)
df0 = inputcsv(0)
print('size of df',df_base.info())
# input fields
start_range = 1 #inclusive
end_range = 2 #exclusive
Testset = 'PS'
Plottype = 'Manually Chosen 20'
IncreasesetPS = ['200%','550%']
IncreasesetTS = ['75%','8.33%']
IncreasesetCS = ['167%', '450%']
palette = sns.color_palette("husl", end_range)
##### Draw 2D scatter plot with only distinct points #####
#Gather  Distinct Data
for i in range(start_range,end_range):
    df1 = inputcsv(i)
    df_wide = compare(df_base,df1)
    df1_distinct,df1_distinct_oldpostemp = extractdistinct(df_wide)
    print('distinct route set{}'.format(i),len(df1_distinct))

"""
README: Uncomment part 1 for running PS set, part 2 for TS set, part 3 for CS
"""
# Part 1
#
# x_axis = 'PhysicalTime'
# y_axis = 'totalTime'
# sns.scatterplot(x='{}'.format(x_axis),y='{}'.format(y_axis),data=df_base,marker='o',color='black',label="Set {} (standard)".format(str(df_base['betaSet'][1])))
#
#
# for i in range(start_range,end_range):
#     df1 = inputcsv(i)
#     df_wide = compare(df_base,df1)
#     df1_distinct,df1_distinct_oldpostemp = extractdistinct(df_wide)
#     sns.scatterplot(x='{}_set{}'.format(x_axis,str(df1['betaSet'][1])), y='{}_set{}'.format(y_axis,str(df1['betaSet'][1])),marker='o', data=df1_distinct, color=palette[i],label="Set {} ({} of standard walking time, bike betas)".format(str(df1['betaSet'][1]),IncreasesetPS[i-1]))
#     df1_distinct_oldpos = calculateED(i,df1_distinct_oldpostemp,x_axis,y_axis)
#     for j in range(0,len(df1_distinct)):
#         drawline(i,df1_distinct,j,x_axis,y_axis,palette[i])
#
# plt.xlabel('{}'.format(x_axis))
# plt.ylabel('{}'.format(y_axis))
#
# plt.legend(bbox_to_anchor=(0.8, -0.14))
# plt.tight_layout()
# plt.title('Beta Sets Comparison: {} - {}'.format(x_axis,y_axis))
# path = 'results/plots/[{0}][{1}]{2}-{3}Beta{4}-Beta{5}.png'.format(Testset,Plottype,x_axis,y_axis,str(df0['betaSet'][1]),str(end_range-1))
# plt.savefig(path, dpi=1000)
# plt.show()

# Part 2
#
# x_axis = 'cost'
# y_axis = 'totalTime'
# sns.scatterplot(x='{}'.format(x_axis),y='{}'.format(y_axis),data=df_base,marker='o',color='black',label="Set {} (standard)".format(str(df_base['betaSet'][1])))
#
#
# for i in range(start_range,end_range):
#     df1 = inputcsv(i)
#     df_wide = compare(df_base,df1)
#     df1_distinct,df1_distinct_oldpostemp = extractdistinct(df_wide)
#     sns.scatterplot(x='{}_set{}'.format(x_axis,str(df1['betaSet'][1])), y='{}_set{}'.format(y_axis,str(df1['betaSet'][1])),marker='o', data=df1_distinct, color=palette[i],label="Set {} ({} of standard cost beta)".format(str(df1['betaSet'][1]),IncreasesetTS[i-1]))
#     df1_distinct_oldpos = calculateED(i,df1_distinct_oldpostemp,x_axis,y_axis)
#     for j in range(0,len(df1_distinct)):
#         drawline(i,df1_distinct,j,x_axis,y_axis,palette[i])
#
# plt.xlabel('{}'.format(x_axis))
# plt.ylabel('{}'.format(y_axis))
# plt.legend(bbox_to_anchor=(0.8, -0.14))
# plt.tight_layout()
#
# plt.title('Beta Sets Comparison: {} - {}'.format(x_axis,y_axis))
# path = 'results/plots/[{0}][{1}]{2}-{3}Beta{4}-Beta{5}.png'.format(Testset,Plottype,x_axis,y_axis,str(df0['betaSet'][1]),str(end_range-1))
# plt.savefig(path, dpi=1000)
# plt.show()

# Part 3
#
# x_axis = 'cost'
# y_axis = 'totalTime'
# sns.scatterplot(x='{}'.format(x_axis),y='{}'.format(y_axis),data=df_base,marker='o',color='black',label="Set {} (standard)".format(str(df_base['betaSet'][1])))
#
#
# for i in range(start_range,end_range):
#     df1 = inputcsv(i)
#     df_wide = compare(df_base,df1)
#     df1_distinct,df1_distinct_oldpostemp = extractdistinct(df_wide)
#     sns.scatterplot(x='{}_set{}'.format(x_axis,str(df1['betaSet'][1])), y='{}_set{}'.format(y_axis,str(df1['betaSet'][1])),marker='o', data=df1_distinct, color=palette[i],label="Set {} ({} of standard cost beta)".format(str(df1['betaSet'][1]),IncreasesetCS[i-1]))
#     df1_distinct_oldpos = calculateED(i,df1_distinct_oldpostemp,x_axis,y_axis)
#     for j in range(0,len(df1_distinct)):
#         drawline(i,df1_distinct,j,x_axis,y_axis,palette[i])
#
# plt.xlabel('{}'.format(x_axis))
# plt.ylabel('{}'.format(y_axis))
# plt.legend(bbox_to_anchor=(0.8, -0.14))
# plt.tight_layout()
#
# plt.title('Beta Sets Comparison: {} - {}'.format(x_axis,y_axis))
# path = 'results/plots/[{0}][{1}]{2}-{3}Beta{4}-Beta{5}.png'.format(Testset,Plottype,x_axis,y_axis,str(df0['betaSet'][1]),str(end_range-1))
# plt.savefig(path, dpi=1000)
# plt.show()




