import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_stud
from metalog import support_func



""" File: metalog """

def fit(x,
        bounds=[0, 1],
        boundedness='u',
        term_limit=13,
        term_lower_bound=2,
        step_len=0.01,
        probs=np.nan,
        fit_method='any',
        save_data=False):
    # Input validation
    
    if ((type(x) is not list) and (type(x) is not np.ndarray)) or (not all(isinstance(b, (int, float, np.int_, np.float_)) for b in x)):
        raise Exception("Error: 'x' must be a numeric vector of type 'list' or 'numpy.ndarray'")
    
    if ((type(bounds) is not list) and (type(bounds) is not np.ndarray)) or (not all(isinstance(b, (int, float, np.int_, np.float_)) for b in bounds)):
        raise Exception("Error: 'bounds' must be a numeric vector of type 'list' or 'numpy.ndarray'")
    
    if term_limit % 1 != 0:
        raise Exception("Error: 'term_limit' parameter should be an integer between 3 and 30")
    
    if term_lower_bound % 1 != 0:
        raise Exception("Error: 'term_lower_bound' parameter should be an integer")
    
    if (np.sum(np.isnan(probs)) == 0) and (len(probs) != len(x)):
        raise Exception("Error: 'probs' vector and 'x' vector must be the same length")
    
    if np.sum(np.isnan(probs)) == 0:
        if not all(isinstance(b, (int, float, np.int_, np.float_)) for b in probs):
            raise Exception("Error: 'probs' parameter should be a numeric vector")
        
        if np.max(probs) > 1 or np.min(probs) < 0:
            raise Exception("Error: 'probs' parameter must have values between (bun not including): 0 and 1")
    
    if len(x) <= 2:
        raise Exception("Error: 'x' must be of length 3 or greater")
    
    if len(bounds) != 2 and boundedness == 'b':
        raise Exception("Error: 'bounds' must have two bounds: upper and lower as a numeric vector (i.e. [12, 45])")
    
    if (np.max(bounds) < np.min(bounds)) and (boundedness == 'b'):
        raise Exception("Error: in 'bounds' upper bound must be greater than lower bound")
    
    if (np.min(x) < np.min(bounds)) and (boundedness == 'b'):
        raise Exception("Error: lower bound in 'bounds' must be less than or equal to the smallest value of 'x'")
    
    if (np.max(bounds) < np.max(x)) and (boundedness == 'b'):
        raise Exception("Error: upper bound in 'bounds' must be greater than or equal to the largest value of 'x'")
    
    if (len(bounds) != 1) and ((boundedness == 'su') or (boundedness == 'sl')):
        raise Exception("Error: 'bounds' must have only one bound")
    
    if boundedness == 'su':
        bounds = np.append(np.min(x), bounds)
    
    if boundedness == 'sl':
        bounds = np.append(bounds, np.max(x))
    
    if boundedness != 'u' and boundedness != 'su' and boundedness != 'sl' and boundedness != 'b':
        raise Exception("Error: 'boundedness' parameter must be: 'u', 'su', 'sl' or 'b' only")
    
    if (np.max(x) > bounds[1]) and (boundedness == 'su'):
        raise Exception("Error: for 'su' in 'boundedness' the upper bound must be greater than or equal to the largest value of 'x'")
    
    if (np.min(x) < bounds[0]) and (boundedness == 'sl'):
        raise Exception("Error: for 'sl' in 'boundedness' the lower bound must be less than or equal to the smalles values of 'x'")
    
    if term_limit < 3 or term_limit > 30:
        raise Exception("Error: 'term_limit' parameter should be an integer between 3 and 30")
    
    if term_limit > len(x):
        raise Exception("Error: 'term_limit' must be less than or equal to the length of the vector 'x'")
    
    if term_lower_bound > term_limit:
        raise Exception("Error: 'term_lower_bound' must be less than or equal to 'term_limit'")
    
    if term_lower_bound < 2:
        raise Exception("Error: 'term_lower_bound' must have a value of 2 or greater")
    
    if step_len < 0.001 or step_len > 0.01:
        raise Exception("Error: 'step_len' must be >= to 0.001 and <= to 0.01")
    
    if fit_method != 'OLS' and fit_method != 'LP' and fit_method != 'any':
        raise Exception("Error: 'fit_method' can only be values: 'OLS', 'LP' or 'any'")
    
    if type(save_data) != bool:
        raise Exception("Error: 'save_data' must be True or False")
    
    
    # Create a dictionary (list) to hold all the objects
    myList = {}
    myList['params'] = {}
    myList['params']['bounds'] = bounds
    myList['params']['boundedness'] = boundedness
    myList['params']['term_limit'] = term_limit
    myList['params']['term_lower_bound'] = term_lower_bound
    myList['params']['step_len'] = step_len
    myList['params']['fit_method'] = fit_method
    myList['params']['number_of_data'] = len(x)
    myList['params']['save_data'] = save_data
    
    # This stores the original data for later use when bayesian updating
    if save_data == True:
        myList['params']['original_data'] = x
    
    # Handle the probabilities --- this also converts x as pandas data frame
    if np.sum( np.isnan(probs) ) != 0:
        x = support_func.MLprobs(x, step_len=step_len)
        # x = pd.DataFrame({'x': MLprobs(x, step_len=step_len)})
    else:
        x = pd.DataFrame({'x': x, 'probs': probs})
    
    # Build the z vector based on the boundedness
    if boundedness == 'u':
        x['z'] = x['x']
    
    if boundedness == 'sl':
        x['z'] = np.log(x['x'] - bounds[0])
    
    if boundedness == 'su':
        x['z'] = -np.log(bounds[1] - x['x'])
    
    if boundedness == 'b':
        x['z'] = np.log( (x['x'] - bounds[0]) / (bounds[1] - x['x']) )
    
    myList['dataValues'] = x
    
    # Construct the Y Matrix initial values
    Y = pd.DataFrame({'y1': np.ones(x.shape[0])})
    Y['y2'] = np.log( x['probs'] / (1 - x['probs']) )
    Y['y3'] = (x['probs'] - 0.5) * Y['y2']
    
    if term_limit > 3:
        Y['y4'] = x['probs'] - 0.5
    
    # Complete the values through the term limit
    if term_limit > 4:
        # breakpoint()
        for i in range(5, term_limit+1):
             y = 'y' + str(i)
             if i % 2 != 0:
                 Y[y] = Y['y4'] ** (i // 2)
             if i % 2 == 0:
                 z = 'y' + str(i-1)
                 Y[y] = Y['y2'] * Y[z]
    
    myList['Y'] = Y
    
    # Build a vectors for each term and
    # build the metalog m(pdf) and M(quantile) dataframes
    myList = support_func.a_vector_OLS_and_LP(myList,
                                 term_limit = term_limit,
                                 term_lower_bound = term_lower_bound,
                                 bounds = bounds,
                                 boundedness = boundedness,
                                 fit_method = fit_method,
                                 diff_error = 0.001,
                                 diff_step = 0.001)
    
    # Build the Components for Bayesian Updating
    if save_data == True and term_lower_bound <= 3:
        Y = np.array(myList['Y'])
        gamma = np.matmul(Y.T, Y)
        myList['params']['bayes'] = {}
        myList['params']['bayes']['gamma'] = gamma
        myList['params']['bayes']['mu'] = myList['A']
        
        v = np.array([])
        for i in range(term_lower_bound, term_limit+1):
            v = np.append(v, myList['params']['number_of_data']-i)
        
        a = v / 2
        myList['params']['bayes']['a'] = a
        myList['params']['bayes']['v'] = v
        
        # For now we will just use the 3 term standard metalog
        if len(v) == 1:
            v = np.nan
            a = np.nan
        else:
            v = v[1]
            a = a[1]
        
        # Use the simple 3 term standard form
        s = np.array([0.1, 0.5, 0.9])
        Ys = pd.DataFrame(np.ones(3), columns=['y1'])
        
        # Construct the Y Matrix initial values
        Ys = pd.concat([Ys, pd.DataFrame( np.log(s/(1-s)), columns=['y2'] )], axis=1)
        Ys = pd.concat([Ys, pd.DataFrame( (s - 0.5) * np.array(Ys['y2']), columns=['y3'] )], axis=1)
        q_bar = np.matmul(np.array(Ys), np.array(myList['A']['a3'][0:3]))
        myList['params']['bayes']['q_bar'] = q_bar
        
        my_t_distr = t_stud(v)
        s2 = ( (q_bar[2] - q_bar[1]) / my_t_distr.ppf(0.9) ) ** 2
        gamma = gamma[0:3, 0:3]
        
        # Build the covariance matrix for the students t
        sig = np.matmul(np.matmul(np.array(Ys), np.linalg.inv(gamma)), np.array(Ys).T)
        b = (a * s2) / gamma[1, 1]
        myList['params']['bayes']['sig'] = (b / a) * sig
        myList['params']['bayes']['b'] = b
    
    
    
    
    return(myList)
















""" File: class_method """

def r(m, n=1, term=3):
    
    # Input validation
    valid_terms = m['Validation']['term']
    valid_term_str = [str(i) for i in valid_terms]
    valid_terms_printout = " ".join(valid_term_str)
    
    if type(n) != int or n < 1 or n % 1 != 0:
        raise Exception("Error: 'n' must be a positive integer")
    
    if (type(term) != int) or (term < 2) or (term % 1 != 0) or (term not in valid_terms):
        raise Exception("Error: term must be a single positive integer contained in the metalog object. Available terms are: " + valid_terms_printout)
    
    x = np.random.uniform(size = n)
    Y = pd.DataFrame(np.ones(n), columns=['y1'])
    
    # Counstruct initial Y Matrix values
    Y['y2'] = np.log(x / (1 - x))
    
    if term > 2:
        Y['y3'] = (x - 0.5) * Y['y2']
    
    if term > 3:
        Y['y4'] = x - 0.5
    
    # Complete the values through the term limit
    if term > 4:
        for i in range(5, term+1):
            y = 'y' + str(i)
            if i % 2 != 0:
                Y[y] = Y['y4'] ** (i // 2)
            if i % 2 == 0:
                z = 'y' + str(i - 1)
                Y[y] = Y['y2'] * Y[z]
    
    amat = 'a' + str(term)
    a = m['A'][amat]
    s = np.matmul(np.array(Y), np.array(a)[0:term])
    
    if m['params']['boundedness'] == 'sl':
        s = m['params']['bounds'][0] + np.exp(s)
    
    if m['params']['boundedness'] == 'su':
        s = m['params']['bounds'][1] - np.exp(s)
    
    if m['params']['boundedness'] == 'b':
        s = m['params']['bounds'][0] + m['params']['bounds'][1] * np.exp(s) / (1 + np.exp(s))
    
    return s



def q(m, y, term = 3):
    
    # Input validation
    valid_terms = m['Validation']['term']
    valid_term_str = [str(i) for i in valid_terms]
    valid_terms_printout = " ".join(valid_term_str)
    
    if (type(term) != int) or (term < 2) or (term % 1 != 0) or (term not in valid_terms):
        raise Exception("Error: term must be a single positive integer contained in the metalog object. Available terms are: " + valid_terms_printout)
    
    if ( not all(isinstance(b, (int, float, np.int_, np.float_)) for b in y) ) or (type(y) is not list) or (max(y) >= 1) or (min(y) <= 0):
        raise Exception("Error: 'y' must be a positive numeric vector of 'numpy.ndarray'-type with values between 0 and 1")
    
    y = np.array(y)
    
    Y = pd.DataFrame(np.ones(len(y)), columns = ['y1'])
    
    # Construct the Y Matrix initial values
    Y['y2'] = np.log(y / (1 - y))
    
    if term > 2:
        Y['y3'] = (y - 0.5) * Y['y2']
    
    if term > 3:
        Y['y4'] = y - 0.5
    
    # Complete the values through the term limit
    if term > 4:
        for i in range(5, term+1):
            y = 'y' + str(i)
            if i % 2 != 0:
                Y[y] = Y['y4'] ** (i // 2)
            if i % 2 == 0:
                z = 'y' + str(i - 1)
                Y[y] = Y['y2'] * Y[z]
    
    amat = 'a' + str(term)
    a = m['A'][amat]
    s = np.matmul(np.array(Y), np.array(a[0:term]))
    
    if m['params']['boundedness'] == 'sl':
        s = m['params']['bounds'][0] + np.exp(s)
    
    if m['params']['boundedness'] == 'su':
        s = m['params']['bounds'][1] - np.exp(-s)
    
    if m['params']['boundedness'] == 'b':
        s = (m['params']['bounds'][0] + m['params']['bounds'][1] * np.exp(s)) / (1 + np.exp(s))
    
    return s



def p(m, q, term = 3):
    
    # Input validation
    valid_terms = m['Validation']['term']
    valid_term_str = [str(i) for i in valid_terms]
    valid_terms_printout = " ".join(valid_term_str)
    
    if (type(term) != int) or (term < 2) or (term % 1 != 0) or (term not in valid_terms):
        raise Exception("Error: term must be a single positive integer contained in the metalog object. Available terms are: " + valid_terms_printout)
    
    if not all(isinstance(b, (int, float, np.int_, np.float_)) for b in q):
        raise Exception("Error: 'q' must be a numeric vector of 'numpy.ndarray'-type")
    
    qs = np.array([support_func.newtons_method_metalog(m, q[i], term) for i in range(len(q))])
    return qs



def d(m, q, term = 3):
    
    # Input validation
    valid_terms = m['Validation']['term']
    valid_term_str = [str(i) for i in valid_terms]
    valid_terms_printout = " ".join(valid_term_str)
    
    if (type(term) != int) or (term < 2) or (term % 1 != 0) or (term not in valid_terms):
        raise Exception("Error: term must be a single positive integer contained in the metalog object. Available terms are: " + valid_terms_printout)
    
    if not all(isinstance(b, (int, float, np.int_, np.float_)) for b in q):
        raise Exception("Error: 'q' must be a numeric vector of 'numpy.ndarray'-type")
    
    qs = np.array([support_func.newtons_method_metalog(m, q[i], term) for i in range(len(q))])
    ds = np.array([support_func.pdfMetalog_density(m=m, y=qs[i], t=term) for i in range(len(qs))])
    
    return ds










""" Summary, Plot and Update """

def summary(m):
    print(' -----------------------------------------------\n',
      'SUMMARY OF METALOG DISTRIBUTION OBJECT\n',
      '-----------------------------------------------\n'
      '\nPARAMETERS\n', '\n',
      'Term Limit: ', m['params']['term_limit'], '\n',
      'Term Lower Bound: ', m['params']['term_lower_bound'], '\n',
      'Boundedness: ', m['params']['boundedness'], '\n',
      'Bounds (only used based on boundedness): ', m['params']['bounds'], '\n',
      'Step Length for Distribution Summary: ', m['params']['step_len'], '\n',
      'Method Use for Fitting: ', m['params']['fit_method'], '\n',
      'Number of Data Points Used: ', m['params']['number_of_data'], '\n',
      'Original Data Saved: ', m['params']['save_data'], '\n',
      '\n\nVALIDATION AND FIT METHOD\n', '\n',
      m['Validation'])



def plot(m, norm=True):
    
    # Collecting data to set limits of axes
    res_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound']) \
                                                     + ' Terms', len(m['M'].iloc[:, 0])),
                                   'pdfValues': m['M'].iloc[:, 0],
                                   'quantileValues': m['M'].iloc[:, 1],
                                   'cumValue': m['M']['y']
                                   })
    if m['M'].shape[-1] > 3:
        for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
            temp_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound'] + i - 1) \
                                                          + ' Terms', len(m['M'].iloc[:, 0])),
                                        'pdfValues': m['M'].iloc[:, i * 2 - 2],
                                        'quantileValues': m['M'].iloc[:, i * 2 - 1],
                                        'cumValue': m['M']['y']})
            res_data = pd.concat([res_data, temp_data], ignore_index=True)
    
    # Collecting data into dictionary
    InitialResults = {}
    InitialResults[str(m['params']['term_lower_bound']) + ' Terms'] = pd.DataFrame({
            'pdfValues': m['M'].iloc[:, 0],
            'quantileValues': m['M'].iloc[:, 1],
            'cumValue': m['M']['y']
            })
    
    if m['M'].shape[-1] > 3:
        for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
            InitialResults[str(m['params']['term_lower_bound'] + i - 1) + ' Terms'] = pd.DataFrame({
                    'pdfValues': m['M'].iloc[:, i * 2 - 2],
                    'quantileValues': m['M'].iloc[:, i * 2 - 1],
                    'cumValue': m['M']['y']
                    })
    
    # ggplot style
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(len(InitialResults), 2, figsize=(8, 3*len(InitialResults)), sharex='col')
    
    for i in range(2, len(InitialResults) + 2):
        # Plotting PDF
        ax[i-2, 0].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['pdfValues'],
              linewidth=2)
        if norm:
            ax[i-2, 0].axis([min(res_data['quantileValues']), max(res_data['quantileValues']),
                  min(res_data['pdfValues']), max(res_data['pdfValues'])])
        
        if i != len(InitialResults) + 1:
            ax[i-2, 0].set(title=str(i) + ' Terms', ylabel='PDF')
        else:
            ax[i-2, 0].set(title=str(i) + ' Terms', ylabel='PDF', xlabel='Quantiles')
        
        
        # Plotting CDF
        ax[i-2, 1].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['cumValue'],
              linewidth=2)
        ax[i-2, 1].axis([min(res_data['quantileValues']), max(res_data['quantileValues']),
              min(res_data['cumValue']), max(res_data['cumValue'])])
        if i != len(InitialResults) + 1:
            ax[i-2, 1].set(title=str(i) + ' Terms', ylabel='CDF')
        else:
            ax[i-2, 1].set(title=str(i) + ' Terms', ylabel='CDF', xlabel='Quantiles')

    
    plt.tight_layout()
    plt.show()



def update(m, new_data):
    if ((type(new_data) is not list) and (type(new_data) is not np.ndarray)) or (not all(isinstance(b, (int, float, np.int_, np.float_)) for b in new_data)):
        raise Exception("Error: 'new_data' must be a numeric vector of type 'list' or 'numpy.ndarray'")
    
    if m['params']['save_data'] == False:
        raise Exception("Error: your metalog object does not contain saved data. If you want to update your disritubiton you must set save_data = TRUE in the original distribution creation.")
    
    all_data = np.append(new_data, m['params']['original_data'])
    
    updated_metalog = fit(all_data,
                          bounds = m['params']['bounds'],
                          boundedness = m['params']['boundedness'],
                          term_limit = m['params']['term_limit'],
                          term_lower_bound = m['params']['term_lower_bound'],
                          step_len = m['params']['step_len'],
                          probs = np.nan,
                          fit_method = m['params']['fit_method'],
                          save_data = True)
    
    Y = np.array(updated_metalog['Y'])
    gamma = np.matmul(Y.T, Y)
    updated_metalog['params']['bayes'] = {}
    updated_metalog['params']['bayes']['gamma'] = gamma
    updated_metalog['params']['bayes']['mu'] = updated_metalog['A']
    
    v = np.array([])
    for i in range(updated_metalog['params']['term_lower_bound'], updated_metalog['params']['term_limit']+1):
        v = np.append(v, updated_metalog['params']['number_of_data']-i)
    
    a = v / 2
    updated_metalog['params']['bayes']['a'] = a
    updated_metalog['params']['bayes']['v'] = v
    
    # For now we will just use the 3 term standard metalog
    if len(v) == 1:
        v = np.nan
        a = np.nan
    else:
        v = v[1]
        a = a[1]
    
    # Use the simple 3 term standard form
    s = np.array([0.1, 0.5, 0.9])
    Ys = pd.DataFrame(np.ones(3), columns=['y1'])
    
    # Construct the Y Matrix initial values
    Ys = pd.concat([Ys, pd.DataFrame( np.log(s/(1-s)), columns=['y2'] )], axis=1)
    Ys = pd.concat([Ys, pd.DataFrame( (s - 0.5) * np.array(Ys['y2']), columns=['y3'] )], axis=1)
    q_bar = np.matmul(np.array(Ys), np.array(updated_metalog['A']['a3'][0:3]))
    updated_metalog['params']['bayes']['q_bar'] = q_bar
    
    gamma = gamma[0:3, 0:3]
    
    # Build the covariance matrix for the students t
    sig = np.matmul(np.matmul(np.array(Ys), np.linalg.inv(gamma)), np.array(Ys).T)
    
    b = 0.5 * updated_metalog['params']['square_residual_error'][len(updated_metalog['params']['square_residual_error'])-1]
    
    updated_metalog['params']['bayes']['sig'] = (b / a) * sig
    updated_metalog['params']['bayes']['b'] = b
    
    return updated_metalog