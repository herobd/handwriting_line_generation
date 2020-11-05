
#{0: [[1,'gen','auto'],[1,'disc']]
class Curriculum:

    def __init__(self,lesson_desc):
        self.lessons=[]
        self.valid = set() #all elements, as we are doing this just to measure error
        self.eval = set()
        self.need_sep_gen_opt = False
        self.need_sep_style_ex_opt = False
        self.need_style_in_disc = False
        self.sample_disc=False
        if lesson_desc==0:
            self.lessons=[]
        else:
            for iteration, lessons in lesson_desc.items():
                lessons_expanded = []
                for lesson in lessons:
                    dup=1
                    new_lesson = []
                    for a in lesson:
                        if type(a) is str:
                            if 'auto-style' in a:
                                self.need_sep_gen_opt = True
                            if 'style-ex-only' in a:
                                self.need_sep_style_ex_opt = True
                            if 'style-super' in a:
                                self.need_style_in_disc = True
                            if 'sample-disc' in a:
                                self.sample_disc=True
                            new_lesson.append(a)
                            if 'gen' not in a and 'disc' not in a and a!='split-style' and 'triplet' not in a: #as GAN losses aren't too informative...
                                self.valid.add(a)
                            if 'disc' not in a and a!='split-style' and 'triplet' not in a: 
                                self.eval.add(a)
                        elif type(a) is int:
                            dup=a
                        else:
                            raise ValueError('unknown thing in lessons: {}'.format(a))
                    for i in range(dup):
                        lessons_expanded.append(new_lesson)
                self.lessons.append( (int(iteration),lessons_expanded) )

        #self.lessons.sort(lambda a,b: b[0]-a[0]) #reverse sort based on iteration
        self.lessons.sort(key=lambda a: a[0], reverse=True)
        self.valid = list(self.valid)
        self.valid.append('valid')
        self.eval = list(self.eval)
        self.eval.append('eval')

    def getLesson(self,iteration):
        while len(self.lessons)>0 and iteration>=self.lessons[-1][0]:
            self.current_lessons = self.lessons.pop()[1]

        return self.current_lessons[ iteration%len(self.current_lessons) ]

    def getValid(self):
        return self.valid
    def getEval(self):
        return self.eval
