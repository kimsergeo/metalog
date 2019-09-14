import numpy as np
import pandas as pd
from scipy.optimize import linprog





""" File: support """

# Supporting functions called inside the metalog function call

# Build the quantiles through a base function
def MLprobs(x, step_len):
    
    l = len(x)
    
    # Calculate the liklihood as an interpolation
    probs = np.zeros(l)
    for i in range(l):
        if i == 0:
            probs[i] = 0.5 / l
        else:
            probs[i] = probs[i-1] + (1 / l)
    
    # If the data is very long we down convert to a smaller but representative
    # vector using the step_len default is 0.01 which is a 109 element vector with
    # fine values in the tail (tailstep)
    if len(x) > 100:
        y = np.arange(step_len, 1-step_len+0.0000000001, step_len)
        tailstep = step_len / 10
        
        y = np.append(np.append(np.arange(tailstep, np.min(y), tailstep),
                     y), 
                     np.arange(np.max(y)+tailstep, np.max(y) + 9.001 * tailstep, tailstep))
        
        if y[0] == 0 or y[0] == 0.:
            y = np.delete(y, 0)
        elif y[-1] == 1 or y[-1] == 1.:
            y = np.delete(y, -1)
            
        x = np.quantile(x, y)
        probs = y
        
    # Gather data in Pandas Data Frame
    x = pd.DataFrame({'x': x, 'probs': probs})
    
    # Returns Pandas Data Frame
    return x



def pdfMetalog(a, y, t, bounds = [], boundedness = 'u'):
    d = y * (1 - y)
    f = (y - 0.5)
    l = np.log(y / (1 - y))
    
    # Initiate pdf
    
    # For the first three terms
    x = a[1] / d
    if a[2] != 0:
        x = x + a[2] * ((f / d) + l)
    
    # For the fourth term
    if t > 3:
        x = x + a[3]
    
    # Initialize some counting variables
    e = 1
    o = 1
    
    # For all other terms greater than 4
    if t > 4:
        for i in range(5, t+1):
            if i % 2 != 0:
                # iff odd
                x = x + ((o + 1) * a[i-1] * f**o)
                o = o + 1
            if i % 2 == 0:
                # iff even
                x = x + a[i-1] * ( ((f**(e+1)) / d) + (e + 1) * f**e * l )
                e = e + 1
    
    # Some change of variables here for boundedness
    x = x**(-1)
    
    if boundedness != 'u':
        M = quantileMetalog(a, y, t, bounds = bounds, boundedness = 'u')
    
    if boundedness == 'sl':
        x = x * np.exp(-M)
    
    if boundedness == 'su':
        x = x * np.exp(M)
    
    if boundedness == 'b':
        x = (x * (1 + np.exp(M)) ** 2) / ( (bounds[1] - bounds[0]) * np.exp(M) )
    
    # Returns a single float
    return x



def quantileMetalog(a, y, t, bounds = [], boundedness = 'u'):
    # Some values for calculation
    f = y - 0.5
    l = np.log(y / (1 - y))
    
    # For the first three terms
    x = a[0] + a[1] * l + a[2] * f * l
    
    # For the fourth term
    if (t > 3):
        x = x + a[3] * f
    
    # Some tracking variables
    o = 2
    e = 2
    
    # For all other terms greater than 4
    if t > 4:
        for i in range(5, t+1):
            if i % 2 == 0:
                x = x + a[i-1] * f**e * l
                e += 1
            if i % 2 != 0:
                x = x + a[i-1] * f**o
                o += 1
    
    if boundedness == 'sl':
        x = bounds[0] + np.exp(x)
    
    if boundedness == 'su':
        x = bounds[1] - np.exp(-x)
    
    if boundedness == 'b':
        x = (bounds[0] + bounds[1] * np.exp(x)) / (1 + np.exp(x))
    
    # Returns a single float
    return x



# Function for returning the matrix of differentiation terms
def diffMatMetalog(term_limit, step_len):
    
    y = np.arange(step_len, 1-step_len+0.0000000001, step_len)
    Diff = np.array([])
    
    for i in range(len(y)):
        d = y[i] * (1 - y[i])
        f = (y[i] - 0.5)
        l = np.log(y[i] / (1 - y[i]))
        
        # Initiate pdf
        diffVector = 0
        
        # For the first three terms
        x = 1 / d
        diffVector = np.append(diffVector, x)
        
        if term_limit > 2:
            diffVector = np.append(diffVector, f / d + l)
        
        # For the fourth term
        if term_limit > 3:
            diffVector = np.append(diffVector, 1)
        
        # Initialize some counting variables
        e = 1
        o = 1
        
        # For all other terms greater than 4
        if term_limit > 4:
            for j in range(5, term_limit+1):
                if j % 2 != 0:
                    # iff odd
                    diffVector = np.append(diffVector, (o+1) * f**o)
                    o += 1
                if j % 2 == 0:
                    # iff even
                    diffVector = np.append(diffVector, f**(e+1) / d + (e + 1) * f**e * l)
                    e += 1
        
        if i == 0:
            Diff = np.hstack([Diff, diffVector])
        else:
            Diff = np.vstack([Diff, diffVector])
    
    new_Diff = np.array([])
    for i in range(Diff.shape[1]):
        if i == 0:
            new_Diff = np.hstack([new_Diff,  Diff[:, i]])
            new_Diff = np.vstack([new_Diff, -Diff[:, i]])
        else:
            new_Diff = np.vstack([new_Diff,  Diff[:, i]])
            new_Diff = np.vstack([new_Diff, -Diff[:, i]])
    
    new_Diff = new_Diff.T
    
    # new_Diff is multidimensional array (matrix)
    return new_Diff



def newtons_method_metalog(m, q, term):
    # A simple newtons method application
    alpha_step = 0.01
    err = 0.0000001
    temp_err = 0.1
    y_now = 0.5
    
    avec = 'a' + str(term)
    a = m['A'][avec]
    i = 1
    
    while temp_err > err:
        frist_function = quantileMetalog(a, y_now, term, m['params']['bounds'], m['params']['boundedness']) - q
        derv_function = pdfMetalog(a, y_now, term, m['params']['bounds'], m['params']['boundedness'])
        y_next = y_now - alpha_step * frist_function * derv_function
        temp_err = abs(y_next - y_now)
        
        if y_next > 1:
            y_next = 0.99999
        if y_next < 0:
            y_next = 0.000001
        
        y_now = y_next
        i += 1
        
        if i > 10000:
            print('Approximation taking too long, quantile value: ', q, ' is too far from distribution median. Try plot() to see distribution.')
    
    # Returns a single float
    return y_now



def pdfMetalog_density(m, t, y):
    
    avec = 'a' + str(t)
    a = m['A'][avec]
    bounds = m['params']['bounds']
    boundedness = m['params']['boundedness']
    
    d = y * (1 - y)
    f = y - 0.5
    l = np.log(y / (1 - y))
    
    # Initiate pdf
    
    # For the first three terms
    x = a[1] / d
    if a[2] != 0:
        x = x + a[2] * (f/d + l)
    
    # For the fourth term
    if t > 3:
        x = x + a[3]
    
    # Initialize some counting variables
    e = 1
    o = 1
    
    # For all other terms greater than 4
    if t > 4:
        for i in range(5, t+1):
            if i % 2 != 0:
                # iff odd
                x = x + (o + 1) * a[i-1] * f**o
                o += 1
            if i % 2 == 0:
                # iff even
                x = x + a[i-1] * ( ((f**(e + 1)) / d) + (e + 1) * (f**e) * l )
                e += 1
    
    # Some change of variables here for boundedness
    x = x**(-1)
    
    if boundedness != 'u':
        M = quantileMetalog(a, y, t, bounds=bounds, boundedness='u')
    
    if boundedness == 'sl':
        x = x * np.exp(-M)
    
    if boundedness == 'su':
        x = x * np.exp(M)
    
    if boundedness == 'b':
        x = (x * (1 + np.exp(M))**2) / ((bounds[1] - bounds[0]) * np.exp(M))
    
    # Returns a single float
    return(x)















""" File: a_vector """

def a_vector_OLS_and_LP(myList,
                        term_limit,
                        term_lower_bound,
                        bounds,
                        boundedness,
                        fit_method,
                        diff_error = 0.001,
                        diff_step = 0.001):

    # Some place holder values
    A = np.array([])
    c_a_names = np.array([])
    c_m_names = np.array([])
    Mh = np.array([])
    Validation = pd.DataFrame({})
    
    for i in range(term_lower_bound, term_limit+1):
        Y = myList['Y'].loc[:, 'y1':'y'+str(i)]
        z = myList['dataValues']['z']
        y = myList['dataValues']['probs']
        step_len = myList['params']['step_len']
        methodFit = 'OLS'
        a = 'a' + str(i)
        m_name = 'm' + str(i)
        M_name = 'M' + str(i)
        c_m_names = np.hstack([c_m_names, m_name, M_name])
        c_a_names = np.hstack([c_a_names, a])
        
        # Try to use the OLS approach
        try:
            temp = np.matmul( 
                    np.matmul( 
                            np.linalg.inv( np.matmul(np.array(Y).T, np.array(Y)) ), 
                            np.array(Y).T ), 
                    np.array(z) )
            OLS_error = False
        except:
            OLS_error = True
        
        if OLS_error:
            temp = a_vector_LP(myList,
                               term_limit = i,
                               term_lower_bound = i,
                               diff_error = diff_error,
                               diff_step = diff_step)
            methodFit = 'Linear Program'
        
        temp = np.append(temp, np.zeros(term_limit - i))
        
        # Build a 'y' vector for smaller data sets
        if len(z) < 100:
            y = np.arange(step_len, 1-step_len+0.0000000001, step_len)
            tailstep = step_len / 10
            y = np.append( np.append( np.arange(tailstep, np.min(y)-tailstep+0.0000000001, tailstep), y ), 
                          np.arange(np.max(y)+tailstep, np.max(y) + 9.001 * tailstep, tailstep) )
        
        # Get the list and quantile values back for validation
        tempList = pdf_quantile_builder(temp, y=y, term_limit=i, bounds=bounds, boundedness=boundedness)
        
        # If it is not a valid PDF run and the OLS version was used the LP version
        if tempList['valid'] == 'no' and fit_method != 'OLS':
            temp = a_vector_LP(myList, term_limit = i, term_lower_bound = i,
                               diff_error = diff_error, diff_step = diff_step)
            temp = np.append(temp, np.zeros(term_limit - i))
            methodFit = 'Linear Program'
            
            # Get the list and quantile values back for validation
            tempList = pdf_quantile_builder(temp, y = y, term_limit = i,
                                            bounds = bounds, boundedness = boundedness)
        
        if len(Mh) != 0:
            Mh = np.vstack([Mh, tempList['m']])
            Mh = np.vstack([Mh, tempList['M']])
        
        if len(Mh) == 0:
            Mh = np.hstack([Mh, tempList['m']])
            Mh = np.vstack([Mh, tempList['M']])
            
        if len(A) != 0:
            A = np.vstack([A, temp])
        
        if len(A) == 0:
            A = np.hstack([A, temp])
        
        tempValidation = pd.DataFrame({'term': i, 'valid': tempList['valid'], 'method': methodFit}, index=[i])
        Validation = pd.concat([Validation, tempValidation])
    
    Mh = Mh.T
    A = A.T
    
    A = pd.DataFrame(A, columns=c_a_names)
    Mh = pd.DataFrame(Mh, columns=c_m_names)
    a1 = pd.DataFrame({'a1': np.ones(len(A))})
    
    # Calculate the error on the data values
    A = pd.concat([a1, A], axis=1)
    Est = np.matmul( np.array(myList['Y']), np.array(A) )
    Z = np.array([list(myList['dataValues']['z']),] * A.shape[1]).T
    
    myList['params']['square_residual_error'] = np.sum(((Z - Est) ** 2).T, axis=1)
    
    myList['A'] = A
    myList['M'] = Mh
    myList['M']['y'] = tempList['y']
    myList['Validation'] = Validation
    
    return myList



def a_vector_LP(myList,
                term_limit,
                term_lower_bound,
                diff_error = 0.001,
                diff_step = 0.001):
    
    cnames = np.array([])
    for i in range(term_lower_bound, term_limit+1):
        Y = myList['Y'].loc[:, 'y1':'y'+str(i)]
        z = myList['dataValues']['z']
        
        # Building the objective function using abs value LP formulation
        Y_neg = -Y
        new_Y = pd.concat([Y.iloc[:,0], Y_neg.iloc[:,0]], axis=1)
        
        for j in range(1, len(Y.iloc[0, :])):
            new_Y = pd.concat([new_Y, Y.iloc[:, j]], axis=1)
            new_Y = pd.concat([new_Y, Y_neg.iloc[:, j]], axis=1)
        
        a = 'a' + str(i)
        cnames = np.append(cnames, a)
        
        # Building the constraint matrix
        error_mat = np.array([])
        
        for k in range(1, len(Y)+1):
            front_zeros = np.zeros(2 * (k - 1))
            ones = np.array([1, -1])
            trail_zeros = np.zeros(2 * ( len(Y.iloc[:,0]) - k ))
            
            if k == 1:
                error_vars = np.hstack([ones, trail_zeros])
            elif k != 1:
                error_vars = np.hstack([front_zeros, ones, trail_zeros])
            
            if k == 1:
                error_mat = np.hstack([error_mat, error_vars])
            else:
                error_mat = np.vstack([error_mat, error_vars])
        
        new = np.hstack([error_mat, new_Y])
        diff_mat = diffMatMetalog(i, diff_step)
        diff_zeros = np.array([])
        
        for v in range(len(diff_mat[:, 0])):
            zeros_temp = np.zeros(2 * len(Y.iloc[:, 0]))
            if v == 0:
                diff_zeros = np.hstack([diff_zeros, zeros_temp])
            else:
                diff_zeros = np.vstack([diff_zeros, zeros_temp])
        
        diff_mat = np.hstack([diff_zeros, diff_mat])
        
        # Objective function coeficients
        f_obj = np.hstack([ np.ones(2 * len(Y.iloc[:, 0])), np.zeros(2 * i)])
        
        # Coeficients of Equality constraints
        f_eq_coef = new
        
        # Right-hand side for Equality constraints
        f_eq_rhs = np.array(z)
        
        # Coeficients of Inequality constraints
        f_ineq_coef = -diff_mat
        
        # Right-hand side for Inequality constraints
        f_ineq_rhs = -np.repeat(diff_error, len(diff_mat[:, 0]))
        
        # Solving the linear program
        lp_sol = linprog(c=f_obj,
                         A_ub=f_ineq_coef, b_ub=f_ineq_rhs,
                         A_eq=f_eq_coef, b_eq=f_eq_rhs,
                         method='revised simplex')
        
        # Consolidating solution back into the vector
        tempLP = lp_sol['x'][ (2 * len(Y.iloc[:, 0])) : len(lp_sol['x'])]
        temp = np.zeros(int(len(tempLP) / 2))
        
        for r in range(1, int(len(tempLP) / 2) + 1):
            temp[r-1] = tempLP[2*r-2] - tempLP[2*r-1]
    
    # 'temp' is numpy array
    return temp















""" File: pdf_quantile_functions """

def pdf_quantile_builder(temp, y, term_limit, bounds, boundedness):
    myList = {}
    
    # Build PDF
    m = np.array([pdfMetalog(temp, y[0], term_limit, bounds=bounds, boundedness=boundedness)])
    
    for j in range(1, len(y)):
        tempPDF = pdfMetalog(temp, y[j], term_limit, bounds=bounds, boundedness=boundedness)
        m = np.append(m, tempPDF)
    
    # Build quantile values
    M = np.array([quantileMetalog(temp, y[0], term_limit, bounds=bounds, boundedness=boundedness)])
        
    for j in range(1, len(y)):
        tempQuant = quantileMetalog(temp, y[j], term_limit, bounds=bounds, boundedness=boundedness)
        M = np.append(M, tempQuant)
    
    # Add trailing and leading zero's for pdf bounds. As well as y
    if boundedness == 'sl':
        m = np.append(0, m)
        M = np.append(bounds[0], M)
        y = np.append(0, y)
    
    if boundedness == 'su':
        m = np.append(m, 0)
        M = np.append(M, bounds[1])
        y = np.append(y, 1)
    
    if boundedness == 'b':
        m = np.append(np.append(0, m), 0)
        M = np.append(np.append(bounds[0], M), bounds[1])
        y = np.append(np.append(0, y), 1)
    
    myList['m'] = m
    myList['M'] = M
    myList['y'] = y
    
    # PDF validation
    myList['valid'] = pdfMetalogValidation(myList['m'])
    
    return myList



# PDF validation function
# Call this feasibility
def pdfMetalogValidation(x):
    y = np.min(x)
    
    if y >= 0:
        return 'yes'
    elif y < 0:
        return 'no'