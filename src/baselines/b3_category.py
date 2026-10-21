"""B3: empirical subcategory probability, category fallback only if unavailable."""

from collections import Counter

from src.baselines.base import BaseRecommender, ScoringContext
from src.data_loader.mind_reader import NewsMetadata


class CategoryBaseline(BaseRecommender):
    name = "B3_category"

    def __init__(self, news: dict[str, NewsMetadata]):
        self.news = dict(news)
        self.diagnostics: Counter = Counter()

    def score(self, context: ScoringContext) -> tuple[float, ...]:
        categories, subcategories = Counter(), Counter()
        stats = self.diagnostics
        stats["impressions"] += 1
        stats["empty_history_impressions"] += not context.history_ids
        stats["history_items"] += len(context.history_ids)
        for news_id in context.history_ids:
            item = self.news.get(news_id)
            if item is None:
                stats["missing_history_metadata_items"] += 1
                continue
            stats["missing_history_category_items"] += item.category is None
            stats["missing_history_subcategory_items"] += item.subcategory is None
            if item.category is not None:
                categories[item.category] += 1
            if item.subcategory is not None:
                subcategories[item.subcategory] += 1
        category_total, subcategory_total = sum(categories.values()), sum(subcategories.values())
        stats["no_usable_profile_impressions"] += not (category_total or subcategory_total)
        scores = []
        for news_id in context.candidate_ids:
            item = self.news.get(news_id)
            stats["candidate_items"] += 1
            stats["missing_candidate_metadata_items"] += item is None
            stats["missing_candidate_category_items"] += item is None or item.category is None
            stats["missing_candidate_subcategory_items"] += item is None or item.subcategory is None
            if item is not None and item.subcategory is not None and subcategory_total:
                score = subcategories[item.subcategory] / subcategory_total
            elif item is not None and item.category is not None and category_total:
                stats["category_fallback_candidates"] += 1
                score = categories[item.category] / category_total
            else:
                stats["no_profile_or_metadata_candidates"] += 1
                score = 0.0
            scores.append(score)
        return tuple(scores)
