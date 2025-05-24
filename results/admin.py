from django.contrib import admin
from .models import IntrusionResult

@admin.register(IntrusionResult)
class IntrusionResultAdmin(admin.ModelAdmin):
    list_display = ('row_index', 'attack_cat', 'mse', 'result', 'timestamp')
    list_filter = ('attack_cat', 'result')
    search_fields = ('row_index', 'attack_cat')
