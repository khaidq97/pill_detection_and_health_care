
class PostProcessing(object):
    def __init__(self):
        pass 

    def hard_postprocessing(self, pill_ids, pres_result):
        pres_ids = []
        for data in pres_result:
            if data['label']=='drugname':
                pres_ids.extend(data['ids'])
        pres_ids = list(set(pres_ids))

        if len(pres_ids):
            ids = [idx if idx in pres_ids else '107' for idx in pill_ids]
        else:
            ids = pill_ids

        return ids