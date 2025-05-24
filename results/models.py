from django.db import models
from django.utils import timezone
from datetime import datetime


class IntrusionResult(models.Model):
    row_index = models.IntegerField()
    attack_cat = models.CharField(max_length=100)
    mse = models.FloatField()
    result = models.CharField(max_length=50)
    ct_src_dport_ltm = models.CharField(max_length=50, default='N/A')
    rate = models.CharField(max_length=50, default='N/A')
    dwin = models.CharField(max_length=50, default='N/A')
    dload = models.CharField(max_length=50, default='N/A')
    swin = models.CharField(max_length=50, default='N/A')
    ct_dst_sport_ltm = models.CharField(max_length=50, default='N/A')
    ct_state_ttl = models.CharField(max_length=50, default='N/A')
    sttl = models.CharField(max_length=50, default='N/A')
    timestamp = models.DateTimeField(default=timezone.now)
    src = models.GenericIPAddressField( null=True, blank=True)
    proto = models.CharField(max_length=20, default='N/A')
    state = models.FloatField(default=0.0)





    def __str__(self):
        return f"Row {self.row_index} - {self.attack_cat} - {self.mse} - {self.result} - {self.ct_src_dport_ltm} - {self.rate} - {self.dwin} - {self.dload} - {self.swin} - {self.ct_dst_sport_ltm} - {self.ct_state_ttl} - {self.sttl}  -  {self.timestamp} - {self.src} - {self.proto} - {self.state} " 