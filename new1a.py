# -*- coding: utf-8 -*-
"""
env sbert
"""

from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-mpnet-base-v2')

import numpy as np

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from sentence_transformers.util import cos_sim

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, \
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer

ctx_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

question_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')




str1 = '''

Defect management
Purpose & objectives
A defect management process is needed to guide decision-making to do the right things at the right time and to manage all defects towards resolution effectively and efficiently.

Requirements for defect management
The following are mandatory in defect management:

Document the defect management process in the master test plan.

The IT initiatives must follow the standard defect management workflow and severity classifications.

The defect management process must be communicated to and agreed with the IT initiative's key stakeholders.

The IT initiatives must use a Nordea-approved defect management tool.

All deviations from expected results as described in the test case must be recorded as defects in the defect management tool.

 
Standard defect management flow


This is a standard tool for independent flow, which sets minimum steps to be followed in all development initiatives. Development initiatives can add more steps in addition to the mandatory flow if they are considered useful in the project. 



Tool specific flows are defined in section on guidelines.

Standard defect severity
#

Severity

Description

S1

Critical

The defect prevents   successful completion of one or more critical business processes for all or   most business users or customers. There is no workaround to achieve the expected result.

S2

Major

The defect prevents successful completion of one or more critical business processes for all or most business users or customers. However, there is workaround to achieve the expected result that can be used temporarily.

S3

Minor

The defect prevents successful completion of one or more non-critical business processes for some users, or critical business processes that are rarely used by a very limited number of business users or customers.

S4

Cosmetic

The defect does not prevent successful completion of any business process. It’s related to some inconvenience in use of system functionality or errors in the look and feel of the application or spelling mistakes.

Guides
'''

str2 = '''

Exceptional need for production data


At any time, for any testing activities, it is mandatory to use synthetic or masked data.

Usage of production data on any of  non production environments in the initiative makes the initiative non compliant with Nordea Test Strategy.



If there is a strong reason for using the production data, then following steps are required:

Step 1: Get approval from Test Data Management  team  to load  production data into any non production environment  ( by raising ticket for data consulting in ITSSP).  Test Data Management team will assess the impact of using production data in the non production environments and its connections. 

Step 2: Get authorization from the department governing data privacy (CSO)

Step 3: Get approval from the application owner (the information owner)

Step 4: Get approval by the group operational risk team and Data Protection Office (DPO).

Step 5: Create a risk record in IT Risk Tool by following the Risk Acceptance Technology Information and Security (RATIS) instructions

'''
str3= '''
Executive summary
Along with digital transformation, the adoption of agile ways of working and dev-ops are affecting the world of testing. Also, advanced technologies such as robotic process automation (RPA), AI and machine learning are changing how testing is performed. Technology is high on banks’ strategic agendas nowadays, and testing has therefore also become critical to business outcomes and customer satisfaction.

One of the aims of this test strategy for it to support agile ways of working and speed up the delivery chain, while at the same time keeping things simple and high-level. The strategy describes the basics of what to achieve in testing, highlighting the mandatory requirements for testing at Nordea. It is supported by a guide section – with practical instructions on how to do it. These will be continuously updated to support state-of-the-art testing. 

The Nordea test strategy 4.0 combines the Nordea software test strategy 3.0, Test environment strategy 3.0 and Test data strategy 1.1 into one. With this strategy, we want to emphasise the need to adopt greater levels of test automation to ensure fast, agile deliveries, and to conduct non-functional testing to its full extent to ensure even better product quality. 

To broaden the perspective and to provide more information about how testing fits in with other IT processes, a new section - ‘Testing by a third party’ has been added.

There are also new test metrics and KPIs defined, to be used to control and manage the quality of delivery and testing processes across development initiatives. The test documentation section now includes changes at different frequencies, different automation levels and automated reports, and the section on test tools now describes general requirements related to testing tools.

Testing is an activity that is crucial to product success and of interest to every member of the product team. Thus, this strategy has been written not only for testers and test managers, but the entire team.

'''
str4 = '''


Functional Testing
Purpose and objectives
Functional testing is a quality assurance process for testing to determine the functionality of an IT solution.

Functionality is the capability of the IT solution to provide functions that meet stated and implied requirements when the IT solution is used under specified conditions.

Functional testing of a IT solution involves tests that evaluate functions that the system should perform. Functions are tested by feeding them input and examining the output.

Test basis can typically be described as the input material used as the basis for test analysis and design. Examples of test basis are requirements, architecture descriptions, test object design and implementation.

Typical test activities to plan for in functional testing:

Functional requirements may be described in work products such as business requirements
specifications, epics, user stories, use cases, or functional specifications, or they may be undocumented.
The functions are “what” the system should do.
Functional tests should be performed at all test levels (e.g., tests for components may be based on a
component specification), though the focus is different at each level (see section 2.2).
Functional testing considers the behavior of the software, so black-box techniques may be used to derive
test conditions and test cases for the functionality of the component or system (see section 4.2).
The thoroughness of functional testing can be measured through functional coverage. Functional
coverage is the extent to which some functionality has been exercised by tests, and is expressed as a
percentage of the type(s) of element being covered. For example, using traceability between tests and
functional requirements, the percentage of these requirements which are addressed by testing can be
calculated, potentially identifying coverage gaps.

ISTQB

Test type: A group of test activities based on specific test objectives aimed at specific characteristics of a component or system.


Regression testing = change related testing to detect whether defects have been introduced or uncovered in unchanged areas of the IT solution
Retesting = change related testing performed after fixing a defect to confirm that a failure caused by that defect does not reoccur
Smoke testing = a test suite that covers the main functionality of a component or system to determine whether it works properly before planned testing begins.

'''
str5 = '''
Vision 
As One Nordea, we want to drive a better customer experience and faster deliveries. We measure the customer experience and utilise the data for improvements. We ensure operational stability and compliance of the bank’s solutions to ensure Nordea is a trusted partner, while we drive a "built-in quality" culture and ownership.

We take best practices and technologies to build efficiencies in testing embedded in the delivery pipeline. We use automation for fast feedback enabling fact-based decision-making.



The vision is aligned with the Nordea IT strategic themes.


Scope
All IT initiatives
Different way of working – agile, SAFe, iterative, waterfall
All types of changes – software and infrastructural components, in-house developed software and purchased ones (COTS)
All kinds of IT changes (classified according to different aspects)
Applicable to all internal and external resources
Target audience
The intended audience of the Nordea test strategy is all stakeholders involved in IT deliveries
– "Everyone is responsible for testing and quality".
'''

str6 = '''

It is mandatory to consider all the requirements in Nordea Test Strategy. If a requirement can not be applied due to the nature of the application or system and are decided to be out of scope, it is mandatory to provide a valid justification in the IT initiative Master Test Plan.

Implement the Nordea test process by considering automation, re-usability, a risk-based approach and applying continuous improvement practices

The IT initiatives take ownership of incorporating solutions for compliance  requirements that affect the testing area. If there are any obstacles to secure compliance requirements, the IT Initiative must escalate to the next level.

If a fundamental test basis and enablers, as above, are not provided, escalation to the next level is required.

 Test levels
It is mandatory that the IT initiative, in the planning activity, identifies and organises the test process into test levels.

It is mandatory to consider all the test levels described in the Nordea test strategy when defining the test level approaches.

The applicable and non-applicable test levels are listed in the master test plan.

It is mandatory to define how end-to-end testing is performed in the context of the IT initiative.

Test levels that cannot be applied due to the characteristic of the application or system, and are decided to be out of scope, are listed with a justification.

Test level details are specified in level test plans.
 Test types (Functional and Non-functional)
It is mandatory to consider the test types stated in the Nordea test strategy when defining the test approaches per test level. 
The applicable and non-applicable test types are listed in the master test plan.
Test types that are considered as out of scope are documented with a justification.
It is mandatory to have stakeholder commitment to the selected test types.

If non-functional requirements (NFRs) are missing, escalation is mandatory.

 Quality gates and test evidence in production gate
It is mandatory to define the quality gates used and their quality criteria in the master test plan and describe how the IT initiative has implemented them. 
It is mandatory to provide test evidence to change records prior deployment to production, according to production gate requirements.
Emergency change verification process must be documented in Master Test Plan (MTP)  including sign-off procedures. 
All emergency changes must have production environment validation report (e-mail) attached to change ticket including non-production verification report when applicable.
 Defect management
The IT initiative must follow the standard defect management workflow and severity classifications.

The applied defect management workflow must be documented in the master test plan.

The defect management workflow must be communicated to and agreed with the IT initiative's key stakeholders.

All deviations from expected result as described in the test case must be recorded as defects in the defect management tool.

The IT initiatives must use a defect management tool approved to be used in Nordea.

 Test documentation
The master test plan is a document in its own right, and level test plans are either merged to MTP or are documents in their own right.
Release test plans, release test reports, test cases.
Master test reports – mandatory only for all IT development handovers between teams in a project/program closure situation or when handing over to maintenance.
 Test metrics
Implement mandatory QA execution metrics and QA capability maturity KPIs.
 Test automation
Test automation approach must be described in the master test plan.
Test automation ratio/utilisation – mandatory metric.
 Testing tools
It is mandatory to list the tools to be used in the master test plan (testing tools and tools that support testing).
It is mandatory for all IT initiatives to use tools for test management, defect management and automated execution of tests cases. Test tools should be used for:
support of the whole test process
all test levels (when applicable, if not applicable a valid reason must be given in the test documents)
all test types (functional and non-functional)
version handling of all test cases (functional and non-functional, as well as for manual and automated test cases).
 Test environments
Path from development to production via component integration, one integration and one pre-prod to ensure coherent configurations and that sufficient testing is performed.
All layers within a development, component integration, one integration or one pre-prod environment must be linked to same type of test environment (development to development, component integration to component integration, one integration to one integration, one pre-prod to one pre-prod).
 Test data
Masked production data – mandatory  to perform the steps described in the masked production data procedure.

Synthetic test data – mandatory  to perform the steps described in the synthetic data procedure.
It is mandatory to use synthetic or desensitised data at all times, for all testing activities.
It is mandatory to classify the communication with an external system to and from a Nordea test system. 
 Approvals and business sign off
For Nordea test plans and reports it is mandatory:
to identify affected business areas and required roles for sign off
to get the approvals and sign off from identified business representatives
that the business representatives are employed in a Nordea business unit
to present the business approval and sign off evidence in the respective test document (master test plan, release test plan, release test report and master test report).
'''

str7 = '''
Non-functional testing

Purpose and objectives
Non-functional testing aims to minimise the risk of failure for non-functional quality attributes of the applications, products and solutions. It is the testing of a software application or system for its non-functional requirements (NFRs) – the way a system operates – rather than for the specific behavior of that system.

The objective is to substantially contribute to Nordea's customer experience excellence and operational stability.
Non-functional testing evaluates characteristics of systems and software such as usability, performance efficiency or security. Non-functional testing is the testing of “how well” the system behaves* and how the user experience the system.

Non-functional testing must be considered and executed as early as possible at all test levels, in the same way as functional testing. This means that it must be considered to be applied from the earliest testing levels and all the way through to acceptance testing.

For example, non-functional testing answers questions like:

[Performance] - How fast does this web-page load account data to the screen?
[Reliability] - How often is it not possible to log in to a system due to unplanned downtime?
[Security] - Can I trust that no unauthorized person can access my private data?
[Usability] - How easy is it to use and understand the user interface?

The fulfilment of the non-functional requirements (NFRs) is verified through the non-functional tests. For example, it is ensured that security controls cannot be bypassed, and that hardening of the environment has been performed.

If non-functional requirements (NFRs) are lacking, an escalation should be initiated.

At Nordea the non-functional requirements are classified by using the FURPS+ model (based on the quality attributes). The non-functional testing types are derived from the FURPS+ classification and are grouped in to the following main categories:

In the event of mandatory non-functional tests being omitted, a valid reason and justification must be provided in the master test plan (MTP).

Usability Testing
Testing to determine the extent to which the software product is understood, easy to learn, easy to operate and attractive to the users under specified conditions.

'''

str8='''
 Accessibility testing
Performance testing
Aims to verify the system’s performance requirements such as response time, transactional throughput and number of concurrent users support under a particular workload. It is used to measure the end-to-end performance of a system and to build the confidence in the IT solution before going live.

Performance testing is required to verify the application’s response for the intended number of users, its maximum load-resisting capacity, the application’s capacity for handling the number of transactions required for a given period by the business and the stability under expected and unexpected user load.

Performance test types listed in order of priority:


 1. Baseline testing
Serves as “snapshot” of system performance at a given acceptable load, and forms the basis of comparison with subsequent tests. It is expected that system performance after code change, bug fixes or for a new release, must be better or inline with baseline testing. Application or IT solution is approved to go-live based on baseline results comparison.

This test results forms the base for other release testing and is used to compare the performance after application changes with a known standard of references. For example, application X supports 1000 anticipated users with NFR response times, then the results become baseline. Any changes in application X should lead to better or equivalent performance compared to baseline results. In case the changes shows better performance, the new results will become baseline for next releases.

How it is performed

Same as load testing type. Load test results become baseline test results when system shows better or inline with NFR requirements.

Advantages:

It ensures consistent or better performance for every release.
It ensures system is inline with NFRs.


 2. Load testing
A type of performance testing conducted to evaluate the behavior of a component or system with expected (anticipated) load. For example, numbers of parallel users and/or total numbers of transactions per second, as well as error rate to determine that the load can be handled by the component or system.

Regression testing will be performed using this test type.
No parameters or settings are changed during regression testing. Any changes are to be reviewed and approved by IT initiative.
Approved results from load testing will become baseline/benchmark for next releases/changes.
How it is performed

Detailed NFRs are collected from various stakeholders. A workload model, including identified scenarios, will be designed and approved . A load test will be executed using approved scenario.

Advantages:

It ensures that the IT solution works as expected in production.
It ensures user experience is within NFR limits.
It ensures that the IT solution is in line with NFRs.


 3. Stress testing
A type of performance testing conducted to evaluate an IT solution/component at or beyond the limits of its anticipated or specified workloads, or with reduced availability of resources such as access to memory or servers.

How it is performed

Starting with load testing scenario, system is loaded with various unexpected load points like background jobs, increased endpoint hit rate, adding rendezvous points at various functionalities. During the stress test, other tests like fail-over, load balancer failure etc. can be performed to understand the stress condition/behavior.

Advantages:

It ensures that the IT solution can handle unpredicted loads.
It helps to understand risks and plan mitigations when unpredictable load arrives.
It provides metrics for future costs involved.
'''
str9 ='''

 4. Scalability testing
Testing to determine the scalability of the software product. Scalability is the capability of the software product to be upgraded to accommodate increased loads.

How it is performed

Starting with load testing scenario (referred as 1x), IT solution is loaded with various incremental loads (2x, 3x , 4x,.....). Thereby the IT solution is monitored for various parameters like CPU, memory, database IOPS, disk utilization, disk IO etc. A detailed analysis will be created based on the observations.

Advantages:

It ensures that the IT solution can handle future sales requirements.
It helps to understand when costing need to be reviewed.
It provides insight on any architectural changes, if required.


 5. Volume testing
Testing where the IT solution is subjected to large volumes of data. For example, database size such as million rows in a table, or processing of huge interface files (XML or JSON).

How it is performed

Similar to load testing, but pumping large volume of data into target table of database, or calling interfaces with huge size of data. The interactions can be reading and/or writing on to/from file.

Advantages:

It identifies load issues, when unpredictable data received thru interfaces.
It identifies database IO operations or table locks when processing large volume of data.

 6. Endurance testing
A type of performance testing conducted to evaluate the behavior of a component/IT solution with expected (anticipated) load for longer durations (for example 24 hrs to 1 week), to determine how the IT solution behaves in longer run, and to identify memory leaks.

How it is performed

Similar to load testing but running test execution starting from 24hrs to 1 week.

Advantages:

It ensures that there are no no bottlenecks when running longer duration.
It ensures that there are no memory leaks causing IT solution catastrophic failure.
It identifies required maintenance window for system restarts.
'''
str10='''

Reliability testing
Reliability defines for example the accuracy of system calculations, availability and the system's recoverability.

Reliability is the property of the application to perform without failure for a long period of time, and independent of external influences. Reliability tests are conducted to test the stability and consistency of the application at any given point in time.

For reliability testing, the Nordea resilience framework can be used as a guide.

The term ‘resilience’ refers to the ability of the business to adapt and respond to risks, as well as opportunities, in order to maintain continuous business operations, be a more attractive partner, and enable growth.

Reliability test types listed in order of priority:

 1. Reliability testing
The ability of the software product to perform its required functions under stated conditions for a specified period of time, or for a specified number of operations.

How it is performed

Similar to endurance test with predefined goals and duration. Goals are identified and agreed. For example, a goal can be probability of failure or length of failure.

Advantages:

To identify pattern of repeating failures.
To identify probability of failure on certain conditions.
To identify fixes for possible know failures.


 2. Failover testing
Testing by simulating failure modes or actually causing failures in a controlled environment. Following a failure, the failover mechanism is tested to ensure that data is not lost or corrupted, and that any agreed service levels are maintained (e.g. function availability or response times).


'''

str11='''

 3. Disaster/recovery testing
Testing to determine, regardless of the circumstances, the recoverability of an IT solution when a disaster occured.

How it is performed

Similar to load test with predefined load and duration, various manual controlled operations such as network disconnect, system shutdown and/or load balancer disconnect are performed. Aim is to study the behavior and validate that the IT solution performs predefined rules like auto scaling, traffic redirection etc. when interruptions occurs.

Advantages:

To identify preventive steps that reduce the risk of man-made disaster
To verify corrective measures that restore lost data and working as expected when recovery procedure performed.
To identify any potential outrage due to disaster.


 4. Recoverability testing
Testing to determine impact of system behavior after recovery. There is a slight difference compared to D/R testing. A recovery can be forced failure of the software in a variety of ways. Recovery testing is basically done in order to check how fast and better the application can recover against any type of crash or hardware failure etc.. Examples of such failures are restart of service after crash in predefined time, restart of pod after failure.

How it is performed

Similar to load test with predefined load and duration, various manual controlled operations such as network disconnect, system shutdown and/or load balancer disconnect are performed. Aim is to study the behavior and validate that the IT solution performs predefined rules like auto scaling, traffic redirection etc. when interruptions occurs.

Advantages:

To identify preventive steps that reduce the risk of unknow software crashes
To study end user experience during crash.
To validate browser session after crash.
Security testing
In accordance and in compliance with various local and international regulations for protecting customer and banking data (for example GDPR), security requirements are of paramount priority in the development of an IT initiative.

Applications and interfaces in scope and frequency of security testing
Security test must be performed at least every year and for all externally reachable applications and interfaces as well as all applications/systems that has at least two critical ratings in the Risk Impact score* for either Confidentiality, Integrity or Availability (CIA). If there are changes applied to the application security test must also be performed before the application goes in to production unless the change is considered by Cyber Security and/or CSO to be minor.
Gateways between internal and external system must be security tested to the same extent as an Internet facing application/server.

Applications that have one critical Risk Impact Score for Confidentiality, Integrity or Availability must be security tested at least every three years.

For new applications (in scope for mandatory secuirty testing), the security testing must be executed, and critical as well as high classified findings (see CVSS scoring) remediated before deployment to production.

The requirements on security testing are described in Guidelines on Security testing, and Guidelines on Group Information Security Instructions (GISI), in chapter 4.10.2.3 System Testing and security review.

* available in Mega-Hopex
Security Test Certificate
All applications in scope for mandatory security testing must have a valid security test certificate issued by the Security Testing Team or a valid dispensation for deployment to production. If the application doesn't have a valid security test certificate or a valid dispensation , deployment to production will be rejected in the change management process. The security test certificate and dispensation are documented in Security Test Application (STA) tool. For more information and how to obtain and prolong the security test certificate, visit this page.
'''

str12='''
Findings (aka. vulnerabilities)
All findings detected during the security testing, must be recorded as defects in a defect management tool (e.g. JIRA) for traceability, follow up and transparency. All open defects (including security findings), must be reported in the release test report (RTR). Following is mandatory when creating defects:

Due to the sensitivity of this information details are not allowed to be written directly into the defect, therefore a reference to the security report is required! 
Major (high) or Critical findings will be automatically registered as incidents.
The time frame for fixing defects is (based on the CVSS criticality classification):
30 days for critical defects
60 days for high defects
If the identified vulnerability cannot be mitigated in the software, Application Provider must, in agreement with Application Owner, apply appropriate compensating controls to remediate risks. For more information on possible compensating controls, please read Guidelines on Security testing , chapter 4.8 Alternative mitigation.

If remediation is not possible within the above defined time frame or appropriate compensating controls implemented a Risk Acceptance has to be written. After the risk acceptance has been signed, it has to be reviewed and recommended/not recommended by CSE RATIS (Risk Acceptance Technology, Information and Security).

 CVSS - Common Vulnerability Scoring System
Common Vulnerability Scoring System (CVSS) is an open industry standard for assessing and scoring the severity of computer system security vulnerabilities. The CVSS scoring is mapped to five different severity classes, which are mapped to the Nordea defect severity classification as follows:

9,0 - 10	Critical	Critical
7 - 8,9	High	Major
4 - 6,9	Medium	Minor
0,1 - 3,9	Low	Cosmetic
0	None	N/A
Read more about CVSS in the Guidelines on Security testing.

How to order mandatory security testing
The mandatory security testing can be executed as external or internal security testing, and is coordinated by Security Testing Team. For more information on how the external and internal security testing is interconnected through the security certification process, and how to obtain and prolong the security test certificate, read following documents:

Process description - Security test with external supplier and/or
Process description - Security Test with internal security tester.
Maintainability (Supportability) Testing
Testing to determine the maintainability of a software product

 Portability testing
Testing to determine the portability of a software product. (Portability testing is the process of determining the degree of ease or difficulty to which a software component or application can be effectively and efficiently transferred from one hardware, software or other operational or usage environment to another.)
'''

str13='''
ISTQB

Non-functional testing is testing performed to evaluate that a component or system complies with non-functional requirements.

Guide

Functional and non-functional testing in Nordea Test Framework
Links

For more information on IT security at Nordea, please visit the following pages:

IT security processes
Security Portal
Information Security Documents
Guidelines on Security testing
Guidelines on Group Information Security Instructions (GISI)
For more information on security testing and how to initiate/order, please visit the following pages:

Security Test Certificates
Cyber Security Service Catalogue
How to order Penetration Test
External Security Testing (EST)
Internal Security Testing (IST)
FAQ external penetration testing
For more information on the security control gate (as part of the change process), please visit:

Change Requests for Application Changes/Deployment to Production (security approvals team)
Non-functional requirements

At Nordea, the requirements, are classified by using the FURPS+ model (based on the quality attributes):
'''

str14 = '''
Nordea Test Strategy
Welcome to Nordea's Test Strategy
Nordea Test Strategy provides all the requirements on testing at Nordea and constitutes the strategy for test. You will also find all the supportive instructive materials for testing in the strategy pages.

The test strategy is based on international best practice standards such as ISTQB and SAFe, incorporates external regulations on compliance and testing, and adheres to Nordea's IT strategy. The Nordea test strategy includes requirements on testing methodology, test environments and test data.

Introduction
Documentation & Metrics
Mandatory Items Summary
Enablers
Test Process
Nordea Test Framework
 Version history
4.0.03	December 23rd 2020	
Page Mandatory items:
Removed non-functional testing section
Test types, added following bullets
It is mandatory to consider all non-functional categories with its respective test types:

performance testing
security testing
reliability testing
maintainability testing
usability testing.
If non-functional requirements (NFRs) are missing, escalation is mandatory.

Updated structure fot Test types
Added new page Functional testing
Moved Non-functional testing under Test types
Moved content from page Test types to the two sub-pages Functional testing & Non-functional testing
Page Non-functional testing
Updated the section Security testing according to the updated requirements available in Guidelines on Security testing, and Guidelines on Group Information Security Instructions (GISI)
Clarified description for the non-functional test types performance, reliability, maintainability and usability testing.
January 26th 2021: Moved performance guide test text to Nordea Test Framework (one sentence, not a mandatory requirement).
'''
str15='''
4.0.02	October 21st 2020	
Layout: Improved layout for page level 1 and 2
Page Summary mandatory items:
Clarification of generic requirements regards mandatory

Page Test data management:

Clarified/simplified existing requirements

Moved procedure descriptions to NTF to simplify NTS

One additional requirement

Page Defect management:

Clarified defect work flow

Clarified/simplified existing requirements

One additional requirement

Page Quality gate and test evidence in production gate:

Two additional requirements regards testing of emergency fixes

Page Non-functional testing:

Security testing: Clarified existing requirements

4.0.01	November 5th 2019	
Updates on the following pages:
'''

str16='''
Test environments are a key enabler in securing a great customer experience, stable, reliable production and ongoing simplification efforts within the bank.

The purpose of this section is to provide a clear and common definition of what is required from test environments in order to have pre-conditions for effective testing in place at Nordea.

Business challenges
 Increased impact of insufficient quality
Lack of quality has a direct impact on customer experiences, and quality needs to be driven as early as possible.
Test environments must enable driving quality as early as possible.
An approach that identifies defects early in the development process needs to be promoted.
Agility for delivering faster time-to-market.
Increased competition that delivers reliable, high-quality solutions with good user experience challenges the ambition levels of the bank.
 High rate of change that balances speed and agility
Testing is key element for safeguarding a stable and well-received customer experience.
The set-up of the test environments must accommodate the required agility whilst supporting the massive amount of planned changes.
 Risk and compliance
Requirements from the ECB and FSAs require Nordea to have test environments that reflect production in terms of represented volumes, procedures and end-to-end processing.
Increased development and integration by external suppliers or third-party deliveries. More ‘open’ environments to be able to have external parties that integrate with the bank’s services (e.g. start-ups) require Nordea to be in control of the environments.
Vision
 "Test environments are the production for developers and testers and are treated in a production-like manner"
Test environments are the production for our developers and testers and should be treated in a production-like manner. Some test environments will be more production-like than others, but all will have production-like characteristics. Pre-production is the environment closest to production. Therefore, the test environments must have proper SLAs that ensure stable, secure, accessible and reliable environments. This should be done as a service with clear expectations and support from all stakeholders.

'''

str17='''
Nordea Test Strategy purpose and goals
Purpose and Goals
The purpose of the test strategy is to obtain a helicopter view of how to approach testing at Nordea. The objectives of the Nordea test strategy are to: 
Improve the satisfaction of customers and business.
Reduce operational risk and to increase operational stability in systems.
Ensure compliance to both external and internal regulations, policies and processes.
Support Nordea’s future journey and evolving the company maturity.
Ensure that the IT solutions are delivered according to agreed functional and non-functional requirements.
Enable the business to take informed decisions when deciding to release new products or features.
Support fast and lean development, embracing Agile ways of working.
Connect testing with the rest of the SDLC to build high quality products at Nordea.
Ensure we have a scalable test process supporting small frequent changes and large infrequent changes.
The 9 Nordea Test Strategy objectives have been merged into 5 overall goals depicted in this picture.
'''


#contexts2 = [str1, str2, str3, str4, str5, str6,str7, str8, str9,str10, str11,str12,str13, str14,str15,str16,str17,str18,str19,str22,str23,str24,str25]


contexts2 = [str1,str2,str3,str4, str5, str6,str7, str8, str9,str10, str11,str12,str13, str14,str15,str16,str17]
#10 is too long


questions2 = [

    "what is Scalability ?",
" what should one do with findings detected during the security testing ?",
    "what is mandatory for Nordea test reports ?",
    " who is responsible for Test Process - Defect Management ?",
    "what is mandatory for Nordea test plans and reports ?",
    "what is mandatory for Nordea test plans  ?",
    "is testing necessary ?",
    "what test environments must have ?",
    " what is test environments ?",
    "what does testing consists of ?",
     "what should be done in case of missing non-functional tests ?",
    "what to expect from non-functional testing ?",
    " is it mandatory to use common Nordea-approved test tools ? ",
    "what should we do when non-functional requirements are missing ? "
]

xb_tokens = ctx_tokenizer(contexts2, max_length=512, padding='max_length',
                          truncation=True, return_tensors='pt')
xb = ctx_model(**xb_tokens)

modelnamep = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizerp = AutoTokenizer.from_pretrained(modelnamep)
modelp = AutoModelForQuestionAnswering.from_pretrained(modelnamep)


def get_top_answers(possible_starts, possible_ends, input_ids):
    answers = []
    for start, end in zip(possible_starts, possible_ends):
        answer = tokenizerp.convert_tokens_to_string(tokenizerp.convert_ids_to_tokens(input_ids[start:end + 1]))
        answers.append(answer)
    return answers


def answer_question(question, context, topN):
    inputs = tokenizerp.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizerp.convert_ids_to_tokens(input_ids)
    model_out = modelp(**inputs)

    answer_start_scores = model_out["start_logits"]
    answer_end_scores = model_out["end_logits"]

    possible_starts = np.argsort(answer_start_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
    possible_ends = np.argsort(answer_end_scores.cpu().detach().numpy()).flatten()[::-1][:topN]

    # get best answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizerp.convert_tokens_to_string(tokenizerp.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    answers = get_top_answers(possible_starts, possible_ends, input_ids)

    return {"answer": answer, "answer_start": answer_start, "answer_end": answer_end, "input_ids": input_ids,
            "answer_start_scores": answer_start_scores, "answer_end_scores": answer_end_scores, "inputs": inputs,
            "answers": answers,
            "possible_starts": possible_starts, "possible_ends": possible_ends}


for j in range(1):


    xq_tokens = question_tokenizer(questions2, max_length=512, padding='max_length',
                                   truncation=True, return_tensors='pt')

    xq = question_model(**xq_tokens)

    xb.pooler_output.shape, xq.pooler_output.shape

    for i, xq_vec in enumerate(xq.pooler_output):
        probs = cos_sim(xq_vec, xb.pooler_output)
        argmax = torch.argmax(probs)
        print(probs)
        print('QQQQQQQQQQ')
        print(questions2[i])
        print('AAAAAAAAAAA')
        print(contexts2[argmax])
        print('---')

        # get precise answer

        print('Precise answer')
        print('---')

        #        for q in questions_edu:
        q = questions2[i]
        var2 = contexts2[argmax]
        answer_map = answer_question(q, var2, 5)
        print("\n", i, ".Question:", q)

        print("precise Answers:")

        #  [print((index+1)," ) ",ans) for index,ans in  enumerate(answer_map["answers"]) if len(ans) > 0 ]
        #        print(ans) for index,ans in  enumerate(answer_map["answers"]) if len(ans) > 0
        [print(ans) for index, ans in enumerate(answer_map["answers"]) if len(ans) > 0]
    # get precise answer end



print(" You can ask me 5 questions!!!\n\n")

for j in range(5):
    questions2 = input("What is your question? I a waiting  ")
    print(questions2)
    answer_map = answer_question(questions2, var2, 5)
    [print(ans, "--XXXXX---")  for index, ans in enumerate(answer_map["answers"]) if len(ans) > 0]


        # squad on large  model to get precise answer -p
