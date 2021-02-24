'''>>> len(t[3])
    130
    >>> len(t[4])
    219'''
from typing import Dict
from typing import List
def Analy(
    drop_list: list,
    nondrop_list: list)-> dict:

    def count_scene(log_:list)->dict:
        """[
            1 统计场景出现的频次
            ]

        Args:
            log_ (list): [description]

        Returns:
            dict: [description]
        """    
        up_ = 15
        down_ = 3

        count_dict = {}
        count_sample_dict = {}
        for length in range(down_,up_):
            print('Counting length :',length)
            
            count_for_length_x = {}
            count_sample_for_length_x = {}
            
            for series in log_:
                
                mid_compute_dict = {}
                
                for i in range(len(series) -length):
                    str_ = str(series[i])
                    for i_ in range(length-1):

                        str_next = str(series[i+1+i_])
                        str_ = str_ +str_next

                    try:
                        mid_compute_dict[str_]+=1
                    except:
                        mid_compute_dict[str_] = 1
            # print(mid_compute_dict)
                # in a series
                for index,value in mid_compute_dict.items():

                    try:
                        count_for_length_x[index]    += value
                        
                        count_sample_for_length_x[index] +=1
                    except:
                        count_for_length_x[index]     = value

                        count_sample_for_length_x[index] =1

            count_dict[length] = count_for_length_x
            count_sample_dict[length] = count_sample_for_length_x

            
        return count_dict,count_sample_dict
    
    def compute(
        series_num:int,
        count_dict:dict,
        count_sample_dict:dict)->Dict[str,list]:
        """
        para:[
            1 序列数量 :int
            2 场景发生次数 :dict
            3 发生该场景的序列数量 :dict
        ] 
        return [ 
            rate_sample_coverage ,
            avg_item_perSample ]
        """    
        final_result_dict = {}        
        sample_number    = series_num
        #i = 0
        for scene_length ,datas_dict in count_dict.items():
         
            sample_data_dict = count_sample_dict[scene_length]

            for scene__,value__ in datas_dict.items():
                pass
                #i +=1
                #if i%100000==0:print('compute :',i,'scenes')
                count_by_items   = value__
                count_by_samples = sample_data_dict[scene__]
                
                rate_sample_coverage = count_by_samples/sample_number
                if count_by_samples ==0:
                    avg_item_perSample = 0
                else:
                    avg_item_perSample =  count_by_items/count_by_samples          

                final_result_dict[scene__] = [rate_sample_coverage , avg_item_perSample]

        return final_result_dict

    def compare(
        drop_list: list,
        nondrop_list: list,
        coverage_threshold = 10,
        avg_threshold = 5)-> dict:
        pass
        result = {}
        for name,data in {'drop':drop_list,'nondrop':nondrop_list}.items():
            
            count_dict, count_sample_dict = count_scene(data)

            result[name] = compute(
                series_num = len(data),
                count_dict= count_dict,
                count_sample_dict = count_sample_dict )
        
        import json
        json.dump(result,open('compute_result_drop_nondrop.json','w'))
        
        final_gap_dict = {}
        i = 0
        print('dict len :',len(result['drop']))
        for scene,data in result['drop'].items():

            i+=1
            if i%10000==0:print('i :',i)

            coverage_drop = result['drop'][scene][0]
            avg_drop = result['drop'][scene][1]
            try:
                coverage_nondrop = result['nondrop'][scene][0]
                avg_nondrop = result['nondrop'][scene][1]

                gap_coverage = int(abs( coverage_nondrop-coverage_drop)*100)
                gap_avg = int(abs(avg_drop-avg_nondrop))
                if (gap_coverage>= coverage_threshold) or (gap_avg >=avg_threshold):
                        final_gap_dict[scene] = [gap_coverage,gap_avg]
            except:
                    pass
        return final_gap_dict

    gap = compare(nondrop_list= nondrop_list,drop_list= drop_list)
   
    return gap

t = Analy(drop_list=list_droped_series,nondrop_list=list_nondrop_series)
drop =[
    [11,12,13,11,12,13,1],
    [11,12,13,11,12,13,1],
    [11,12,13,11,12,13,1],
    [11,12,13,11,12,13,1]]
nondrop =[
    [21,22,23,24,25,26,1],
    [11,12,13,24,25,26,1],
    [21,22,23,24,25,26,1],
    [11,12,13,24,25,26,1]]


t,s = t_count_scene(data)

x= compute(
    series_num= len(data),
    count_dict=t,
    count_sample_dict= s
)


def Analy():
    pass
    """[
        1 对比辍学与非辍学数据集的频次数据，选出覆盖率差别大的场景
    ]
    """

def part_count_scene(
    logs:dict,
    scene_:str)->int:

    length = int(len(scene_))# /2)
    
    print('Counting scene :',scene_)
    
    
    result_dict = {}
    for name_,log_ in logs.items():
        count_by_samples = 0
        count_by_items = 0
        for series in log_:
        
            control = 0
            #  if c%1000 ==0 :print('already enumerate :',c)
            for i in range(len(series) -length):
                str_ = str(series[i])
                for i_ in range(length-1):
                    
                    str_next = str(series[i+1+i_])
                    str_ = str_ +str_next

                if str_ == scene_:
                    control = 1
                    count_by_items +=1
                
            if control ==1:
                count_by_samples +=1

        result_dict[name_]=[
            len(log_),
            int(count_by_items),
            int(count_by_samples)]
    
    def analy_()->None:  

        def compute(name_:str)->list:
            """
            return [ 
                rate_sample_coverage ,
                avg_item_perSample ]

            """            
            sample_number    = result_dict[name_][0]
            count_by_items   = result_dict[name_][1]
            count_by_samples = result_dict[name_][2]
            
            rate_sample_coverage = count_by_samples/sample_number
            avg_item_perSample =  count_by_items/(count_by_samples+1)          
            return [ 
                rate_sample_coverage ,
                avg_item_perSample ]

        drop_list = compute('drop')
        nondrop_list = compute('nondrop')
        gap_coverage = int(abs( drop_list[0]-nondrop_list[0])*100)
        gap_avg = int(abs(drop_list[1]-nondrop_list[1] ))

        if (gap_coverage>= 5) or (gap_avg >=5):

            print(
                'gap_coverage :',gap_coverage,'%',
                '\ngap_avg :',gap_avg)

    analy_()
    return None

