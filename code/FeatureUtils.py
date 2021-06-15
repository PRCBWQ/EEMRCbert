class EventDescrip:
    def __init__(self,
                 kind:str,
                 kindindex:int,
                 attribList:list
                 ):
        self.kind = kind
        self.kindindex = kindindex
        self.attribList = attribList
class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels


class TriggerFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 offset_ids,
                 labels=None):
        super(TriggerFeature, self).__init__(token_ids=token_ids,
                                             attention_masks=attention_masks,
                                             token_type_ids=token_type_ids,
                                             labels=labels)
        self.offset_ids = offset_ids


class AttributeFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 trigger_loc,
                 trigger_distance=None,
                 labels=None):
        """
        attribution detection use two handcrafted feature：
        1、trigger label： 1 for the tokens which are trigger, 0 for not;
        2、trigger distance: the relative distance of other tokens and the trigger tokens
        """
        super(AttributeFeature, self).__init__(token_ids=token_ids,
                                          attention_masks=attention_masks,
                                          token_type_ids=token_type_ids,
                                          labels=labels)
        self.trigger_loc = trigger_loc
        self.trigger_distance = trigger_distance
