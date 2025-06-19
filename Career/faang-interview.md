# 🚀 FAANG Interview Preparation Guide

## Overview
Comprehensive guide for preparing for technical interviews at FAANG (Facebook/Meta, Amazon, Apple, Netflix, Google) and other top tech companies.

## 🎯 Interview Process Overview

### Typical Interview Pipeline
```python
interview_process = {
    "1_application": {
        "resume_screening": "ATS optimization crucial",
        "referral": "60% higher success rate",
        "online_assessment": "Some companies have coding tests"
    },
    
    "2_phone_screen": {
        "duration": "45-60 minutes",
        "format": "1-2 coding problems",
        "platform": "CoderPad, HackerRank",
        "expectations": "Clean code, communication"
    },
    
    "3_onsite_virtual": {
        "rounds": "4-6 interviews",
        "types": [
            "Coding (2-3 rounds)",
            "System Design (1-2 rounds)",
            "Behavioral (1-2 rounds)",
            "Domain specific (ML/AI roles)"
        ],
        "duration": "45-60 min each"
    },
    
    "4_team_match": {
        "companies": ["Google", "Meta"],
        "process": "Match with specific teams",
        "duration": "1-4 weeks"
    },
    
    "5_offer": {
        "components": ["Base", "Bonus", "Equity", "Sign-on"],
        "negotiation": "Expected and encouraged"
    }
}
```

## 💻 Coding Interview Preparation

### Data Structures Mastery
```python
# Essential Data Structures Implementation

class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def reverse(self):
        """Reverse linked list - Classic interview question"""
        prev = None
        current = self.head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        self.head = prev
        return self.head
    
    def detect_cycle(self):
        """Floyd's cycle detection algorithm"""
        if not self.head:
            return False
        
        slow = fast = self.head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert_helper(self.root, val)
    
    def _insert_helper(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_helper(node.left, val)
        else:
            node.right = self._insert_helper(node.right, val)
        
        return node
    
    def inorder_traversal(self):
        """Iterative inorder traversal using stack"""
        result = []
        stack = []
        current = self.root
        
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            
            current = stack.pop()
            result.append(current.val)
            current = current.right
        
        return result
    
    def level_order_traversal(self):
        """BFS traversal - Very common interview question"""
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.pop(0)
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)
        
        return result
```

### Algorithm Patterns
```python
# Two Pointers Pattern
def two_sum_sorted(nums, target):
    """Two sum in sorted array"""
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

def container_with_most_water(height):
    """Classic two pointer problem"""
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        min_height = min(height[left], height[right])
        area = width * min_height
        max_area = max(max_area, area)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

# Sliding Window Pattern
def longest_substring_without_repeating(s):
    """Sliding window with hashmap"""
    char_index = {}
    max_length = 0
    start = 0
    
    for end in range(len(s)):
        if s[end] in char_index:
            start = max(start, char_index[s[end]] + 1)
        
        char_index[s[end]] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length

# Dynamic Programming Pattern
def coin_change(coins, amount):
    """Classic DP problem - Minimum coins needed"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_increasing_subsequence(nums):
    """DP with binary search optimization"""
    if not nums:
        return 0
    
    dp = []
    
    for num in nums:
        left, right = 0, len(dp)
        
        while left < right:
            mid = (left + right) // 2
            if dp[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(dp):
            dp.append(num)
        else:
            dp[left] = num
    
    return len(dp)

# Backtracking Pattern
def generate_parentheses(n):
    """Generate all valid parentheses combinations"""
    result = []
    
    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)
    
    backtrack("", 0, 0)
    return result

# Graph Algorithms
def number_of_islands(grid):
    """DFS on 2D grid"""
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        
        grid[r][c] = '0'  # Mark as visited
        
        # Check all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)
    
    return islands
```

### Time and Space Complexity
```python
complexity_cheatsheet = {
    "arrays": {
        "access": "O(1)",
        "search": "O(n)",
        "insertion": "O(n)",
        "deletion": "O(n)",
        "space": "O(n)"
    },
    
    "linked_lists": {
        "access": "O(n)",
        "search": "O(n)",
        "insertion": "O(1)",
        "deletion": "O(1)",
        "space": "O(n)"
    },
    
    "hash_tables": {
        "access": "N/A",
        "search": "O(1) average, O(n) worst",
        "insertion": "O(1) average, O(n) worst",
        "deletion": "O(1) average, O(n) worst",
        "space": "O(n)"
    },
    
    "binary_search_trees": {
        "access": "O(log n) average, O(n) worst",
        "search": "O(log n) average, O(n) worst",
        "insertion": "O(log n) average, O(n) worst",
        "deletion": "O(log n) average, O(n) worst",
        "space": "O(n)"
    },
    
    "sorting_algorithms": {
        "quick_sort": "O(n log n) average, O(n²) worst",
        "merge_sort": "O(n log n) all cases",
        "heap_sort": "O(n log n) all cases",
        "counting_sort": "O(n + k) where k is range"
    }
}
```

## 🏗️ System Design Interview

### System Design Framework
```python
system_design_approach = {
    "1_requirements": {
        "functional": [
            "What features to build?",
            "Who are the users?",
            "What's the scale?"
        ],
        "non_functional": [
            "Performance requirements",
            "Availability (99.9%?)",
            "Consistency vs Availability"
        ],
        "constraints": [
            "Budget limitations",
            "Time constraints",
            "Technical constraints"
        ]
    },
    
    "2_capacity_estimation": {
        "traffic": "Requests per second",
        "storage": "Data size calculations",
        "bandwidth": "Network requirements",
        "example": """
        Twitter-like system:
        - 500M users, 50% DAU = 250M
        - Each user: 2 tweets/day = 500M tweets/day
        - Read:Write = 100:1
        - Storage: 280 chars + metadata = ~3KB/tweet
        - Daily storage: 500M * 3KB = 1.5TB/day
        """
    },
    
    "3_system_interface": {
        "apis": [
            "postTweet(user_id, tweet_text, location)",
            "getTweets(user_id, count, max_id)",
            "followUser(user_id, follow_id)"
        ]
    },
    
    "4_data_model": {
        "database_choice": "SQL vs NoSQL",
        "schema_design": "Tables and relationships",
        "indexing": "Query optimization"
    },
    
    "5_high_level_design": {
        "components": [
            "Load Balancer",
            "Application Servers",
            "Database (Master-Slave)",
            "Cache Layer",
            "CDN"
        ]
    },
    
    "6_detailed_design": {
        "deep_dive": "2-3 components in detail",
        "algorithms": "Ranking, recommendation",
        "data_flow": "Request lifecycle"
    },
    
    "7_scale_and_optimize": {
        "techniques": [
            "Horizontal scaling",
            "Database sharding",
            "Caching strategies",
            "Message queues"
        ]
    }
}
```

### Common System Design Problems
```python
# URL Shortener Design
class URLShortenerDesign:
    """
    Requirements:
    - Shorten long URLs
    - Redirect to original URL
    - Analytics
    - Custom aliases
    - 100M URLs/day
    """
    
    def __init__(self):
        self.components = {
            "api_gateway": "Rate limiting, authentication",
            "app_servers": "Stateless, horizontally scalable",
            "url_service": "Generate short URLs",
            "database": "Store URL mappings",
            "cache": "Redis for hot URLs",
            "analytics": "Async processing with Kafka"
        }
    
    def url_generation_algorithm(self):
        """
        1. Counter-based: Simple but predictable
        2. Hash-based: MD5 -> Base62 (collision handling)
        3. Hybrid: Counter + Hash for uniqueness
        """
        approaches = {
            "base62": "0-9, a-z, A-Z = 62 chars",
            "length": "7 chars = 62^7 = 3.5 trillion URLs",
            "custom": "Check availability in DB"
        }
        return approaches
    
    def database_schema(self):
        return """
        URL_Mapping:
        - short_url (PK)
        - long_url
        - created_at
        - expires_at
        - user_id
        
        Analytics:
        - short_url (FK)
        - timestamp
        - ip_address
        - user_agent
        - referrer
        """
    
    def scaling_considerations(self):
        return {
            "database_sharding": "By short_url prefix",
            "cache_strategy": "LRU with 20% hot URLs",
            "cdn": "Global redirects",
            "monitoring": "Prometheus + Grafana"
        }

# Distributed Message Queue Design
class MessageQueueDesign:
    """
    Design a Kafka-like distributed message queue
    """
    
    def requirements(self):
        return {
            "functional": [
                "Publish messages to topics",
                "Subscribe to topics",
                "Message ordering per partition",
                "At-least-once delivery"
            ],
            "scale": {
                "messages": "1M messages/sec",
                "size": "1KB average message",
                "retention": "7 days"
            }
        }
    
    def architecture(self):
        return {
            "producers": "Client libraries with batching",
            "brokers": {
                "role": "Store and serve messages",
                "storage": "Segmented log files",
                "replication": "Leader-follower pattern"
            },
            "consumers": {
                "groups": "Parallel consumption",
                "offset_management": "Commit to Zookeeper/Kafka"
            },
            "zookeeper": {
                "metadata": "Broker registry",
                "leader_election": "Partition leaders",
                "configuration": "Topic configs"
            }
        }
    
    def partition_strategy(self):
        return """
        1. Hash-based: hash(key) % num_partitions
        2. Round-robin: No key provided
        3. Custom: User-defined partitioner
        
        Benefits:
        - Parallel processing
        - Ordered within partition
        - Horizontal scaling
        """
```

## 🗣️ Behavioral Interview

### STAR Method Framework
```python
class BehavioralResponse:
    def __init__(self, situation, task, action, result):
        self.situation = situation
        self.task = task
        self.action = action
        self.result = result
    
    def format_response(self):
        return f"""
        Situation: {self.situation}
        Task: {self.task}
        Action: {self.action}
        Result: {self.result}
        """

# Example responses
leadership_example = BehavioralResponse(
    situation="Our team's technical lead left suddenly during a critical project phase",
    task="Needed to maintain team momentum and meet the deadline",
    action="""
    - Organized daily standups to track progress
    - Distributed responsibilities based on expertise
    - Created documentation for knowledge sharing
    - Coordinated with stakeholders on priority changes
    """,
    result="""
    - Delivered project on time
    - Team efficiency increased by 20%
    - Received recognition from management
    - Became the permanent tech lead
    """
)

conflict_resolution = BehavioralResponse(
    situation="Two senior engineers disagreed on architecture approach",
    task="Resolve conflict and move project forward",
    action="""
    - Scheduled separate 1:1s to understand perspectives
    - Organized technical discussion with pros/cons
    - Proposed hybrid approach incorporating both ideas
    - Set up POC to validate approach
    """,
    result="""
    - Both engineers bought into solution
    - Hybrid approach proved 30% more efficient
    - Improved team collaboration process
    - Documented decision-making framework
    """
)
```

### Common Behavioral Questions
```python
behavioral_questions = {
    "leadership": [
        "Tell me about a time you led a project",
        "Describe mentoring a junior engineer",
        "How do you handle team conflicts?",
        "Example of influencing without authority"
    ],
    
    "problem_solving": [
        "Describe debugging a complex issue",
        "Time you solved an ambiguous problem",
        "How do you prioritize tasks?",
        "Example of innovative solution"
    ],
    
    "failure_growth": [
        "Tell me about a failure",
        "Time you received critical feedback",
        "Describe a missed deadline",
        "How do you handle mistakes?"
    ],
    
    "collaboration": [
        "Working with difficult stakeholders",
        "Cross-functional project experience",
        "Disagreement with manager",
        "Building consensus"
    ],
    
    "amazon_leadership_principles": [
        "Customer Obsession",
        "Ownership",
        "Invent and Simplify",
        "Are Right, A Lot",
        "Learn and Be Curious",
        "Hire and Develop the Best",
        "Insist on the Highest Standards",
        "Think Big",
        "Bias for Action",
        "Frugality",
        "Earn Trust",
        "Dive Deep",
        "Have Backbone; Disagree and Commit",
        "Deliver Results"
    ]
}
```

## 🤖 ML/AI Specific Interviews

### ML System Design
```python
class MLSystemDesign:
    """Framework for ML system design interviews"""
    
    def ml_pipeline_components(self):
        return {
            "data_collection": {
                "sources": "APIs, databases, streams",
                "formats": "Structured, unstructured",
                "volume": "Estimate data size"
            },
            
            "data_preprocessing": {
                "cleaning": "Missing values, outliers",
                "feature_engineering": "Domain-specific features",
                "transformation": "Scaling, encoding"
            },
            
            "model_training": {
                "algorithm_selection": "Based on problem type",
                "hyperparameter_tuning": "Grid search, Bayesian",
                "distributed_training": "Data/model parallelism"
            },
            
            "model_serving": {
                "deployment": "Real-time vs batch",
                "scaling": "Horizontal scaling",
                "latency": "Caching, model optimization"
            },
            
            "monitoring": {
                "metrics": "Accuracy, latency, drift",
                "alerts": "Performance degradation",
                "retraining": "Scheduled vs triggered"
            }
        }
    
    def recommendation_system_design(self):
        """Design Netflix-like recommendation system"""
        return {
            "requirements": {
                "scale": "100M users, 10K items",
                "latency": "<100ms",
                "personalization": "User preferences"
            },
            
            "approaches": {
                "collaborative_filtering": {
                    "user_based": "Similar users",
                    "item_based": "Similar items",
                    "matrix_factorization": "SVD, NMF"
                },
                
                "content_based": {
                    "features": "Genre, actors, director",
                    "similarity": "Cosine, Jaccard",
                    "profile_learning": "User preferences"
                },
                
                "hybrid": {
                    "weighted": "Combine scores",
                    "switching": "Context-based",
                    "mixed": "Present both"
                }
            },
            
            "architecture": {
                "offline": {
                    "spark_jobs": "Daily model training",
                    "feature_store": "User/item features",
                    "model_store": "Version control"
                },
                
                "online": {
                    "api_server": "FastAPI/Flask",
                    "cache": "Redis for hot users",
                    "fallback": "Popular items"
                },
                
                "feedback_loop": {
                    "implicit": "Views, time spent",
                    "explicit": "Ratings, likes",
                    "processing": "Kafka streams"
                }
            }
        }
```

### ML Coding Questions
```python
# Implement k-means clustering
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        
    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign clusters
            clusters = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, clusters)
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
        
        return self
    
    def _assign_clusters(self, X):
        distances = np.zeros((len(X), self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, clusters):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        
        return new_centroids
    
    def predict(self, X):
        return self._assign_clusters(X)

# Implement gradient descent
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """Linear regression using gradient descent"""
    m, n = X.shape
    theta = np.zeros(n)
    costs = []
    
    for _ in range(n_iterations):
        # Predictions
        y_pred = X.dot(theta)
        
        # Cost
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        costs.append(cost)
        
        # Gradients
        gradients = (1/m) * X.T.dot(y_pred - y)
        
        # Update parameters
        theta -= learning_rate * gradients
    
    return theta, costs

# Implement precision, recall, F1
def classification_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

## 📚 Study Resources

### Books
```python
essential_books = {
    "algorithms": [
        "Cracking the Coding Interview - McDowell",
        "Elements of Programming Interviews - Aziz et al.",
        "Algorithm Design Manual - Skiena",
        "Introduction to Algorithms - CLRS"
    ],
    
    "system_design": [
        "Designing Data-Intensive Applications - Kleppmann",
        "System Design Interview - Alex Xu", 
        "Building Microservices - Newman",
        "High Performance Browser Networking - Grigorik"
    ],
    
    "behavioral": [
        "Behavioral Interview Questions - Narasimhan",
        "60 Seconds and You're Hired - Ryan",
        "STAR Interview Method - Clark"
    ],
    
    "ml_specific": [
        "Machine Learning Interviews - Huyen",
        "The ML Interview Book - Washam",
        "Deep Learning - Goodfellow et al."
    ]
}
```

### Online Platforms
```python
practice_platforms = {
    "coding": {
        "LeetCode": {
            "pros": "FAANG-focused problems, company tags",
            "features": "Premium for company-specific",
            "strategy": "Focus on medium problems"
        },
        
        "HackerRank": {
            "pros": "Structured learning paths",
            "features": "Certificates, contests",
            "use_case": "Initial screening prep"
        },
        
        "CodeSignal": {
            "pros": "Real interview environment",
            "features": "Proctored assessments",
            "companies": "Uber, Robinhood use it"
        }
    },
    
    "system_design": {
        "Pramp": "Free peer mock interviews",
        "SystemDesignInterview.com": "Detailed solutions",
        "High Scalability": "Real-world architectures",
        "YouTube": "Success in Tech, Gaurav Sen"
    },
    
    "mock_interviews": {
        "Pramp": "Free peer-to-peer",
        "Interviewing.io": "Anonymous practice",
        "Exponent": "FAANG-specific coaching"
    }
}
```

### Study Schedule
```python
def create_study_plan(weeks_until_interview):
    """Create personalized study plan"""
    
    if weeks_until_interview >= 12:
        return {
            "weeks_1_4": {
                "focus": "Data structures fundamentals",
                "daily": "2-3 easy problems",
                "topics": ["Arrays", "Strings", "LinkedLists", "Trees"]
            },
            "weeks_5_8": {
                "focus": "Algorithms and patterns",
                "daily": "2 medium problems",
                "topics": ["DP", "Graphs", "Backtracking"]
            },
            "weeks_9_10": {
                "focus": "System design",
                "practice": "2 designs per week",
                "study": "DDIA book, YouTube"
            },
            "weeks_11_12": {
                "focus": "Mock interviews",
                "frequency": "3-4 per week",
                "review": "Weak areas"
            }
        }
    
    elif weeks_until_interview >= 4:
        return {
            "week_1": "Top 75 LeetCode problems",
            "week_2": "Company-specific problems",
            "week_3": "System design crash course",
            "week_4": "Mock interviews + behavioral"
        }
    
    else:
        return {
            "priority": "Company-specific prep",
            "daily": "3-5 problems from target company",
            "mock": "Daily mock interviews",
            "review": "Previous interview feedback"
        }
```

## 💰 Negotiation Tips

### Compensation Components
```python
compensation_breakdown = {
    "base_salary": {
        "negotiable": "10-20% typically",
        "factors": ["Experience", "Location", "Level"],
        "research": ["Levels.fyi", "Blind", "Glassdoor"]
    },
    
    "equity": {
        "types": ["RSUs", "Options", "Profit sharing"],
        "vesting": "4 years with 1-year cliff typical",
        "negotiable": "20-50% more common"
    },
    
    "bonus": {
        "signing": "Most negotiable component",
        "annual": "Target percentage of base",
        "performance": "Company-dependent"
    },
    
    "benefits": {
        "health": "Usually standard",
        "pto": "Sometimes negotiable",
        "remote": "Increasingly negotiable"
    }
}

negotiation_strategy = {
    "1_get_competing_offers": "Strongest position",
    "2_know_your_worth": "Research market rates",
    "3_negotiate_everything": "Not just base salary",
    "4_be_patient": "Don't accept immediately",
    "5_get_it_in_writing": "Email confirmation",
    "6_consider_total_comp": "4-year view"
}
```

---

*Success is where preparation meets opportunity. Good luck with your FAANG interviews!* 🚀💼