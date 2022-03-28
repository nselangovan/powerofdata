Student Database (MongoDB)
Here is the student dataset in the json format.
Perform the following operation:
First create a database and then load the student.json dataset.
Insert the students record into the collection.
Queries need to answer:

1)      Find the student name who scored maximum scores in all (exam, quiz and homework)?


        db.students.aggregate([
    {
        '$set': {
            'total_score': {
                '$sum': '$scores.score'
            }
        }
    }, {
        '$sort': {
            'total_score': -1, 
            'posts': 1
        }
    },
    {
         $limit : 1 
         }
])

        
2)      Find students who scored below average in the exam and pass mark is 40%?     


db.students.find({ "$and": [{"scores.0.score":{"$lt":48.67}},{"scores.0.score":{"$gte":40}}]}).count()


3)      Find students who scored below pass mark and assigned them as fail, and above pass mark as pass in all the categories.


db.students.aggregate
([{$unwind: '$scores'}, {$group: {
 _id: '$_id'
}}, {$project: {
 name: 1,
 scores: 1,
 pass: {
  $cond: {
   'if': {
    $and: [
     {
      $gte: [
       '$scores.0.score',
       40
      ]
     },
     {
      $gte: [
       '$scores.1.score',
       40
      ]
     },
     {
      $gte: [
       '$scores.2.score',
       40
      ]
     }
    ]
   },
   then: true,
   'else': false
  }
 }
}}])


4)      Find the total and average of the exam, quiz and homework and store them in a separate collection.


db.students.aggregate([
        { $unwind: "$scores" },
        { $group : { _id: "$scores.type", average : {  $avg : "$scores.score" }, total: {$sum:"$scores.score"} } },
        { $out: { db: "school", coll: "tot_avg" } }
    ])


5)      Create a new collection which consists of students who scored below average and above 40% in all the categories.


    db.students.aggregate([
    {
        '$match': {
            '$and': [
                {
                                '$and': [
                {
                    'scores.0.score': {
                        '$gte': 40
                    }
                }, {
                    'scores.0.score': {
                        '$lt': 48.67367075950175
                    }
                }
            ]
            }, 
                {
                                '$and': [
                {
                    'scores.1.score': {
                        '$gte': 40
                    }
                }, {
                    'scores.1.score': {
                        '$lt': 48.99672319430254
                    }
                }
            ]
            }, 
                {
                                '$and': [
                {
                    'scores.2.score': {
                        '$gte': 40
                    }
                }, {
                    'scores.2.score': {
                        '$lt': 67.81869620661149
                    }
                }
            ]
            }
            ]
        }
    }
    ,
        { $out: { db: "school", coll: "below_average" } }
])



6)      Create a new collection which consists of students who scored below the fail mark in all the categories.


    db.students.aggregate([
    {
        '$match': {
            '$and': [
                {
                    'scores.0.score': {
                        '$lt': 40
                    }
                }, {
                    'scores.1.score': {
                        '$lt': 40
                    }
                }, {
                    'scores.2.score': {
                        '$lte': 40
                    }
                }
            ]
        }
    }
    ,
        { $out: { db: "school", coll: "all_fail" } }
])


7)      Create a new collection which consists of students who scored above pass mark in all the categories.


    db.students.aggregate([
    {
        '$match': {
            '$and': [
                {
                    'scores.0.score': {
                        '$gte': 40
                    }
                }, {
                    'scores.1.score': {
                        '$gte': 40
                    }
                }, {
                    'scores.2.score': {
                        '$gte': 40
                    }
                }
            ]
        }
    }
    ,
        { $out: { db: "school", coll: "all_pass" } }
])









