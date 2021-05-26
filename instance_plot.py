import os
import argparse
import numpy as np
import utilities
import pandas as pd

def generate_MTSP(random, local_file, filename, n_customers,m_salesman):
    """
    Generate a MTSP problem following
    
        Bektas, T.: The multiple traveling salesman problem: an overview of 
    formulations and solution procedures. Omega, 34(3) (2006), pp. 209–219..

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    """
    #rng = np.random.RandomState(random)
    
    c_x = rng.rand(n_customers) #产生n_customers个[0,1]之间的随机数
    c_y = rng.rand(n_customers)

    f_x = c_x
    f_y = c_y
    # loc_dir = f'data/display_instance/MTSP/location'
    dataframe = pd.DataFrame({'c_x': c_x, 'c_y': c_y})
    # print(dataframe)
    dataframe.to_csv(local_file ,index=True,sep=',')

    # transportation costs
    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)
    
    #print (trans_costs)
    #trans_costs = np.sqrt(
    #        (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1)) #reshape((-1, 1)使得矩阵重组为n*1


    # write problem
    with open(filename, 'w') as file:
        file.write("Minimize\n obj:")
        
        
        file.write("".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_customers)]))
        #
        file.write("\nSubject To\n")
        
        cnt = 0
        cnt = cnt+1
        
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{1}_{j+1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
            
        cnt = cnt+1
        file.write(f" c{cnt}:" + "".join([f" + 1 x_{j+1}_{1}" for j in range(1,n_customers)]) + f" = {m_salesman}\n")
                  
        cnt = cnt+1
        file.write(f" c{cnt}: u_{1} = 0\n")
        
        nvisit = n_customers/m_salesman +1 
        
        #print(n_customers,m_salesman,nvisit)
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} >= 1\n")
            cnt = cnt+1
            #file.write(f" c{cnt}: u_{i+1} <= {n_customers-1}\n")
            file.write(f" c{cnt}: u_{i+1} <= {nvisit-1}\n")
            

        for i in range(0,n_customers):
            for j in range(0,n_customers):
                cnt = cnt+1
                if i == j:
                    file.write(f" c{cnt}: x_{i+1}_{j+1} = 0\n")
                else:
                    file.write(f" c{cnt}: x_{i+1}_{j+1} >= 0\n")
                    cnt = cnt+1
                    file.write(f" c{cnt}: x_{i+1}_{j+1} <= 1\n")
                    
  
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for j in range(n_customers)]) + f" = 1\n")
            
        for j in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}" for i in range(n_customers)]) + f" = 1\n")
            
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: x_{1}_{i+1} + x_{i+1}_{1} <= 1\n")
        
        for i in range(1,n_customers):
            for j in range(1,n_customers):
                if i == j:
                    continue
                    file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} <= {n_customers-m_salesman-1} \n")
                else:
                    cnt = cnt+1
                    file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} + u_{i+1} - u_{j+1} <= {n_customers-m_salesman-1} \n")
        
        file.write("\nBounds\n")
        #for i in range(n_customers):
        #    file.write(f" 1 <= u_{i+1} <= {n_customers}\n")
        
        #for i in range(n_customers):
        #    for j in range(n_customers):
        #        file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nGenerals\n")
            
        #file.write("\nbinary\n")
        
        for i in range(n_customers):
            for j in range(n_customers):
                file.write(f" x_{i+1}_{j+1} ")
        for i in range(n_customers):
            file.write(f" u_{i+1} ")
            
        #file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))
        file.write("\nEnd")

def generate_MinMax_MTSP(random, local_file, filename, n_customers,m_salesman):
    """
    Generate a generate_MinMax_MTSP problem following
    
  Necula, R., Breaban, M., Raschip, M.: Tackling the Bi-criteria Facet of Multiple Traveling Salesman Problem with Ant Colony Systems, 27th International Conference on Tools with Artificial Intelligence (ICTAI), 9-11 November, Vietri sul Mare, Italy, pp. 873-880, 2015

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    """
    #rng = np.random.RandomState(random)
    
    c_x = rng.rand(n_customers) #产生n_customers个[0,1]之间的随机数
    c_y = rng.rand(n_customers)

    f_x = c_x
    f_y = c_y

    dataframe = pd.DataFrame({'c_x': c_x, 'c_y': c_y})
    # print(dataframe)
    dataframe.to_csv(local_file ,index=True,sep=',')

    #m_salesman = 3

    # transportation costs
    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2)
    
    #print (trans_costs)
    #trans_costs = np.sqrt(
    #        (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1)) #reshape((-1, 1)使得矩阵重组为n*1

    # write problem
    with open(filename, 'w') as file:
        file.write("Minimize\n obj:")
        
        
        file.write(f"1 W_{1}\n")
        
        
        #file.write("".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_customers)]))
        #
        file.write("\nSubject To\n")
        
        cnt = 0
        
        
        for k in range(0,m_salesman):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{1}_{j+1}_{k+1}" for j in range(1,n_customers)]) + f" = {1}\n")
        
        for k in range(0,m_salesman):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{j+1}_{1}_{k+1}" for j in range(1,n_customers)]) + f" = {1}\n")
                  
        cnt = cnt+1
        file.write(f" c{cnt}: u_{1} = 0\n")
        
        nvisit = n_customers/m_salesman +1 
        
        #print(n_customers,m_salesman,nvisit)
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} >= 1\n")
            cnt = cnt+1
            file.write(f" c{cnt}: u_{i+1} <= {n_customers-1}\n")
            #file.write(f" c{cnt}: u_{i+1} <= {nvisit-1}\n")
            
        for i in range(0,n_customers):
            for j in range(0,n_customers):
                for k in range(0,m_salesman): 
                    cnt = cnt+1
                    if i == j:
                        file.write(f" c{cnt}: x_{i+1}_{j+1}_{k+1} = 0\n")
                    else:
                        continue
                        file.write(f" c{cnt}: x_{i+1}_{j+1}_{k+1} >= 0\n")
                        cnt = cnt+1
                        file.write(f" c{cnt}: x_{i+1}_{j+1}_{k+1} <= 1\n")
                    
  
        for j in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}_{k+1}" for i in range(n_customers) for k in range(m_salesman)]) + f" = 1\n")
        
        
        for i in range(1,n_customers):
            cnt = cnt+1
            file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}_{k+1}" for j in range(n_customers) for k in range(m_salesman)]) + f" = 1\n")
            
        
        for j in range(1,n_customers):
            for k in range(0,m_salesman):
                cnt = cnt+1
                file.write(f" c{cnt}:" + "".join([f" + 1 x_{i+1}_{j+1}_{k+1} - 1 x_{j+1}_{i+1}_{k+1}" for i in range(n_customers) ]) + f" = 0\n")
        
        #for i in range(1,n_customers):
        #   cnt = cnt+1
        #    file.write(f" c{cnt}: x_{1}_{i+1} + x_{i+1}_{1} <= 1\n")
        
        for i in range(1,n_customers):
            for j in range(1,n_customers):
                if i == j:
                    continue
                for k in range(m_salesman): 
                    #file.write(f" c{cnt}: {n_customers-m_salesman} x_{i+1}_{j+1} <= {n_customers-m_salesman-1} \n")
                    cnt = cnt+1
                    #file.write(f" c{cnt}: + u_{i+1} - u_{j+1} + {n_customers-m_salesman} "+"(" + "".join([f" + x_{i+1}_{j+1}_{k+1}" for k in range(n_customers)]) +")" + f"<={n_customers-m_salesman-1} \n")
                    file.write(f" c{cnt}: + u_{i+1} - u_{j+1}  "+ "".join([f" + {n_customers-m_salesman} x_{i+1}_{j+1}_{k+1}" for k in range(n_customers)]) + f" <= {n_customers-m_salesman-1} \n")


        
        for k in range(m_salesman):
            cnt = cnt +1 
            file.write(f" c{cnt}:" + "".join([f" + {trans_costs[i, j]} x_{i+1}_{j+1}_{k+1}" for i in range(n_customers) for j in range(n_customers)]) + f"-1 W_{1} <= 0\n")
            
        
        
        file.write("\nBounds\n")
        file.write(f"0 <= W_{1} \n")
        #for i in range(n_customers):
        #    file.write(f" 1 <= u_{i+1} <= {n_customers}\n")
        
        #for i in range(n_customers):
        #    for j in range(n_customers):
        #        file.write(f" 0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nGenerals\n")
            
        #
        for i in range(n_customers):
            file.write(f" u_{i+1} ")
        
        file.write("\nbinary\n")
        
        for i in range(n_customers):
            for j in range(n_customers):
                for k in range(m_salesman): 
                    file.write(f" x_{i+1}_{j+1}_{k+1} ")
                    
        #file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))
        file.write("\nEnd")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['MTSP', 'minmax-mtsp', 'minmax-mtsp_9'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=utilities.valid_seed,
        default=0,
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    if args.problem =='MTSP':
        number_of_customers = 12
        number_of_salesman = 3
        local_files = []
        filenames = []
        ncustomerss = []
        nsalesmans = []

        n = 100
        lp_dir = f'data/display_instance/{args.problem}/display_{number_of_customers}_{number_of_salesman}'
        loc_dir = f'data/display_instance/{args.problem}/location'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        os.makedirs(loc_dir)
        local_files.extend([os.path.join(loc_dir, f'instance_{i+1}.csv') for i in range(n)])
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        for local_file, filename, ncs, nsm in zip(local_files, filenames, ncustomerss, nsalesmans): #zip() 函数用于将可迭代的对象作为参数，
                                                                                        #将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            print(f"  generating file {filename} ...")
            generate_MTSP(rng, local_file, filename, n_customers=ncs, m_salesman=nsm)

        print("done.")        

    elif args.problem =='minmax-mtsp':
        number_of_customers = 12
        number_of_salesman = 3
        local_files = []
        filenames = []
        ncustomerss = []
        nsalesmans = []

        n = 100
        lp_dir = f'data/display_instance/{args.problem}/display_{number_of_customers}_{number_of_salesman}'
        loc_dir = f'data/display_instance/{args.problem}/location'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        os.makedirs(loc_dir)
        local_files.extend([os.path.join(loc_dir, f'instance_{i+1}.csv') for i in range(n)])
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        for local_file,filename, ncs, nsm in zip(local_files,filenames, ncustomerss, nsalesmans): #zip() 函数用于将可迭代的对象作为参数，
                                                                                        #将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            print(f"  generating file {filename} ...")
            generate_MinMax_MTSP(rng, local_file, filename, n_customers=ncs, m_salesman=nsm)

        print("done.")

    elif args.problem =='minmax-mtsp_9':
        number_of_customers = 9
        number_of_salesman = 3
        local_files = []
        filenames = []
        ncustomerss = []
        nsalesmans = []

        n = 100
        lp_dir = f'data/display_instance/{args.problem}/display_{number_of_customers}_{number_of_salesman}'
        loc_dir = f'data/display_instance/{args.problem}/location'
        print(f"{n} instances in {lp_dir}")
        os.makedirs(lp_dir)
        os.makedirs(loc_dir)
        local_files.extend([os.path.join(loc_dir, f'instance_{i+1}.csv') for i in range(n)])
        filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
        ncustomerss.extend([number_of_customers] * n)
        nsalesmans.extend([number_of_salesman] * n)

        for local_file,filename, ncs, nsm in zip(local_files,filenames, ncustomerss, nsalesmans): #zip() 函数用于将可迭代的对象作为参数，
                                                                                        #将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            print(f"  generating file {filename} ...")
            generate_MinMax_MTSP(rng, local_file, filename, n_customers=ncs, m_salesman=nsm)

        print("done.")   